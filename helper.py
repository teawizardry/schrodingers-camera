import string
import numpy as np
import time, datetime
import matplotlib.pyplot as plt
import wandb

class Helper:
    def __init__(self) -> None:
        pass
    
    # Label Helper Functions
    def LabelHelper(label):
        if label < 10:
            digits = list(string.digits)
            return digits[label]
        else: # if label > 9 and label < 36:
            caps = list(string.ascii_uppercase)
            return caps[label-10]

    def SecondLabelHelper(second_label):
        if second_label == 0:
            return 'Digit'
        # elif second_label == 1:
        #     return 'Lowercase'
        else:
            return 'Letter'

    # Add Digit or Letter Classification
    def AddLabel(labels_train):
        second_labels = np.empty_like(labels_train)
        for index, label in enumerate(labels_train):
            if Helper.LabelHelper(label) in string.digits:
                # add digit classification - 0
                second_labels[index] = 0
            else:
                # add letter classification - 1
                second_labels[index] = 1
        return second_labels

    # Data Formatter
    def DeleteLower(images, labels):
        new_images = np.delete(images, np.where(labels>35), axis=0)
        new_labels = np.delete(labels, np.where(labels>35))
        return new_images, new_labels

class MetricLogger():
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"
        self.ep_avg_qs_target_plot = save_dir / "q_target_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        self.ep_avg_qs_target = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []
        self.moving_avg_ep_avg_qs_target = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()


    def log_step(self, reward, loss, q, q_target):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_q_target += q_target
            self.curr_ep_loss_length += 1
        wandb.log({"reward": reward, "loss": loss, "q": q, "q_target": q_target})

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward) #.cpu().numpy()
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
            ep_avg_q_target = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
            ep_avg_q_target = np.round(self.curr_ep_q_target / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        self.ep_avg_qs_target.append(ep_avg_q_target)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_q_target = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        mean_ep_q_target = np.round(np.mean(self.ep_avg_qs_target[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)
        self.moving_avg_ep_avg_qs_target.append(mean_ep_q_target)


        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )
        wandb.log({"Episode": episode, "Step": step, "Mean Reward": mean_ep_reward, "Mean Length": mean_ep_length, "Mean Loss": mean_ep_loss, "Mean Q Value": mean_ep_q, "Mean Target Value": mean_ep_q_target})

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_avg_qs_target"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()