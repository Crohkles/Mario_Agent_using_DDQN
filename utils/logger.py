import numpy as np
import time, datetime
import matplotlib.pyplot as plt


class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # Initialize data lists
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Load existing data if log file exists, otherwise create new one
        self._load_existing_data()

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def _load_existing_data(self):
        """Load existing log data if file exists, otherwise create new log file with header"""
        if self.save_log.exists():
            print(f"Found existing log file: {self.save_log}")
            print("Loading previous training data...")
            
            try:
                with open(self.save_log, "r") as f:
                    lines = f.readlines()
                
                # Skip header line if present
                data_lines = lines[1:] if lines and "Episode" in lines[0] else lines
                
                # Parse existing data
                for line in data_lines:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 7:  # Ensure we have all required columns
                        try:
                            episode = int(parts[0])
                            step = int(parts[1])
                            epsilon = float(parts[2])
                            mean_reward = float(parts[3])
                            mean_length = float(parts[4])
                            mean_loss = float(parts[5])
                            mean_q = float(parts[6])
                            
                            # Reconstruct moving averages (these are the values we logged)
                            self.moving_avg_ep_rewards.append(mean_reward)
                            self.moving_avg_ep_lengths.append(mean_length)
                            self.moving_avg_ep_avg_losses.append(mean_loss)
                            self.moving_avg_ep_avg_qs.append(mean_q)
                            
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Could not parse log line: {line}")
                            continue
                
                print(f"Loaded {len(self.moving_avg_ep_rewards)} previous training records")
                
                # Add session separator
                with open(self.save_log, "a") as f:
                    f.write(f"\n# === New Training Session Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                    
            except Exception as e:
                print(f"Error loading existing log data: {e}")
                print("Starting with fresh log file...")
                self._create_new_log_file()
        else:
            print("No existing log file found. Creating new log file...")
            self._create_new_log_file()
    
    def _create_new_log_file(self):
        """Create a new log file with header"""
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

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

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
            plt.clf()
            # Use the complete moving average data (including historical data)
            moving_avg_data = getattr(self, f"moving_avg_{metric}")
            if moving_avg_data:  # Only plot if we have data
                plt.plot(moving_avg_data, label=f"moving_avg_{metric}")
                plt.title(f"Training Progress: {metric}")
                plt.xlabel("Record Points")
                plt.ylabel(metric.replace("_", " ").title())
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(getattr(self, f"{metric}_plot"))
