import sys
from src.debugger import run_debugger
from utils.logger import generate_html_file, Logger 
from utils.frontend import run_frontend 

if __name__ == '__main__':
    Logger.init_logger(
        "--only-errors" in sys.argv,
        "--keep-errors" in sys.argv
    )

    if "--skip-sampling" not in sys.argv:
        run_debugger()

    generate_html_file()
    run_frontend()
