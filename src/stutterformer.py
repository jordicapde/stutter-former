"""Recipe for training a neural speech separation system on WHAM! and WHAMR!
datasets. The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer-wham.yaml --data_folder /your_path/wham_original
> python train.py hparams/sepformer-whamr.yaml --data_folder /your_path/whamr

The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures.

Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import csv
import os

import numpy as np
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
import torch
import torch.nn.functional as F
import torchaudio
from pesq import pesq
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from speechbrain.utils.metric_stats import MetricStats
from tqdm import tqdm


# Define training procedure
class StutterFormer(sb.Brain):
    def compute_forward(self, noisy, target, stage):
        """Forward computations from the noisy to the estimated clean signal."""

        # Unpack lists and put tensors in the right device
        noisy, noisy_length = noisy
        noisy, noisy_length = noisy.to(self.device), noisy_length.to(self.device)

        # Convert targets to tensor
        target = torch.cat(
            [target[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        ).to(self.device)

        # Enhancement
        encoded_noisy = self.hparams.Encoder(noisy)
        mask = self.modules.masknet(encoded_noisy)

        encoded_noisy = torch.stack([encoded_noisy] * self.hparams.num_spks)
        sep_h = encoded_noisy * mask
        estimated_pred = torch.cat(
            [
                self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = noisy.size(1)
        T_est = estimated_pred.size(1)
        estimated_pred = estimated_pred.squeeze(-1)
        if T_origin > T_est:
            estimated_pred = F.pad(estimated_pred, (0, T_origin - T_est))
        else:
            estimated_pred = estimated_pred[:, :T_origin]

        return estimated_pred, target.squeeze(-1)

    def compute_objectives(self, prediction, target):
        """Computes the si-snr loss"""

        return self.hparams.loss(
            target.unsqueeze(-1), prediction.unsqueeze(-1)
        )

    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list
        noisy = batch.stutter_sig
        target = [batch.speech_sig]

        prediction, target = self.compute_forward(
            noisy, target, sb.Stage.TRAIN
        )
        loss = self.compute_objectives(prediction, target)

        if self.hparams.threshold_byloss:
            th = self.hparams.threshold
            loss_to_keep = loss[loss > th]
            if loss_to_keep.nelement() > 0:
                loss = loss_to_keep.mean()
        else:
            loss = loss.mean()

        if (
            loss < self.hparams.loss_upper_lim and loss.nelement() > 0
        ):  # the fix for computational problems
            loss.backward()
            if self.hparams.clip_grad_norm >= 0:
                torch.nn.utils.clip_grad_norm_(
                    self.modules.parameters(), self.hparams.clip_grad_norm
                )
            self.optimizer.step()
        else:
            self.nonfinite_count += 1
            logger.info(
                "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                    self.nonfinite_count
                )
            )
            loss.data = torch.tensor(0).to(self.device)
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        batch_id = batch.id
        noisy = batch.stutter_sig
        target = [batch.speech_sig]

        with torch.no_grad():
            prediction, target = self.compute_forward(noisy, target, stage)
            loss = self.compute_objectives(prediction, target)

        if stage != sb.Stage.TRAIN:
            self.pesq_metric.append(
                ids=batch_id, predict=prediction.cpu(), target=target.cpu()
            )
            self.stoi_metric.append(
                batch_id, prediction.cpu(), target.cpu(), lens=noisy[1], reduction="batch"
            )

        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(batch_id[0], noisy, target, prediction)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(batch_id[0], noisy, target, prediction)

        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            # Define function taking (prediction, target) for parallel eval
            def pesq_eval(pred_wav, target_wav):
                """Computes the PESQ evaluation metric"""
                psq_mode = "wb" if self.hparams.sample_rate == 16000 else "nb"
                try:
                    return pesq(
                        fs=self.hparams.sample_rate,
                        ref=target_wav.numpy(),
                        deg=pred_wav.numpy(),
                        mode=psq_mode,
                    )
                except Exception:
                    print("pesq encountered an error for this data item")
                    return 0

            self.pesq_metric = MetricStats(
                metric=pesq_eval, n_jobs=1, batch_eval=False
            )
            self.stoi_metric = MetricStats(
                metric=stoi_loss, n_jobs=1, batch_eval=True
            )

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stats = {
                "loss": stage_loss,
                "pesq": self.pesq_metric.summarize("average"),
                "stoi": -self.stoi_metric.summarize("average"),
            }

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Learning rate annealing
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stats,
            )
            if (
                hasattr(self.hparams, "save_all_checkpoints")
                and self.hparams.save_all_checkpoints
            ):
                self.checkpointer.save_checkpoint(meta=stats)
            else:
                self.checkpointer.save_and_keep_only(
                    meta=stats, max_keys=["pesq"],
                )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length withing the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def save_results(self, test_data):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""

        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        all_pesqs = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i", "pesq"]

        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, **self.hparams.dataloader_opts
        )

        with open(save_file, "w") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):

                    # Apply Separation
                    mixture, mix_len = batch.mix_sig
                    snt_id = batch.id
                    targets = [batch.s1_sig, batch.s2_sig]
                    if self.hparams.num_spks == 3:
                        targets.append(batch.s3_sig)

                    with torch.no_grad():
                        predictions, targets = self.compute_forward(
                            batch.mix_sig, targets, sb.Stage.TEST
                        )

                    # Compute SI-SNR
                    sisnr = self.compute_objectives(predictions, targets)

                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [mixture] * self.hparams.num_spks, dim=-1
                    )
                    mixture_signal = mixture_signal.to(targets.device)
                    sisnr_baseline = self.compute_objectives(
                        [mixture_signal.squeeze(-1), None], targets
                    )
                    sisnr_i = sisnr - sisnr_baseline

                    # Compute SDR
                    sdr, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        predictions[0][0].t().detach().cpu().numpy(),
                    )

                    sdr_baseline, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        mixture_signal[0].t().detach().cpu().numpy(),
                    )

                    sdr_i = sdr.mean() - sdr_baseline.mean()

                    # Compute PESQ
                    psq_mode = (
                        "wb" if self.hparams.sample_rate == 16000 else "nb"
                    )
                    psq = pesq(
                        self.hparams.sample_rate,
                        targets.squeeze().cpu().numpy(),
                        predictions[0].squeeze().cpu().numpy(),
                        mode=psq_mode,
                    )

                    # Saving on a csv file
                    row = {
                        "snt_id": snt_id[0],
                        "sdr": sdr.mean(),
                        "sdr_i": sdr_i,
                        "si-snr": -sisnr.item(),
                        "si-snr_i": -sisnr_i.item(),
                        "pesq": psq,
                    }
                    writer.writerow(row)

                    # Metric Accumulation
                    all_sdrs.append(sdr.mean())
                    all_sdrs_i.append(sdr_i.mean())
                    all_sisnrs.append(-sisnr.item())
                    all_sisnrs_i.append(-sisnr_i.item())
                    all_pesqs.append(psq)

                row = {
                    "snt_id": "avg",
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                    "pesq": np.array(all_pesqs).mean(),
                }
                writer.writerow(row)

        logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
        logger.info("Mean SISNRi is {}".format(np.array(all_sisnrs_i).mean()))
        logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
        logger.info("Mean SDRi is {}".format(np.array(all_sdrs_i).mean()))
        logger.info("Mean PESQ {}".format(np.array(all_pesqs).mean()))

    def save_audio(self, snt_id, noisy, target, prediction):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create outout folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # Estimated source
        signal = prediction
        signal = signal / signal.abs().max()
        save_file = os.path.join(
            save_path, "item{}_sourcehat.wav".format(snt_id)
        )
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )

        # Original source
        signal = target[0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_source.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )

        # Mixture
        signal = noisy[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )
