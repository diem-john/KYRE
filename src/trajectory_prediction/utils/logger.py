import datetime

class Logger:
    def __init__(self, project_dir):
        self.project_dir = project_dir

    def log(self, msg: str, level: str = "INFO"):
        """
        Helper for printing logs and saving them into a .log file
        """

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = level.upper()
        formatted = f"[{timestamp}] [{level}] {msg}"

        # ANSI color codes
        RESET = "\033[0m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"

        # Assigns color
        if level == "STATUS":
            colored_output = GREEN + formatted + RESET
        elif level == "ERROR":
            colored_output = RED + formatted + RESET
        elif level == "STAGE":
            colored_output = YELLOW + formatted + RESET
        else:  # INFO (default)
            colored_output = formatted

        # Print to terminal
        print(colored_output)

        # Write plain text (no color codes) to file
        with open(f"{self.project_dir}/pipeline.log", "a") as f:
            f.write(formatted + "\n")