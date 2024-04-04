from flask import Flask, render_template, request
from src.debugger import run_debugger
from utils.logger import LOG_DIR, LOG_RESULT_IMG_DIR, toggle_error, generate_html_file, Logger

app = Flask(__name__, template_folder="../" + LOG_DIR, static_folder="../" + LOG_RESULT_IMG_DIR)
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/toggle-error', methods=['POST'])
def handle_checkbox():
    line_no = int(request.form.get('identifier'))
    is_checked = request.form.get('is_checked') == 'true'
    toggle_error(line_no - 1, is_checked)
    generate_html_file()
    return {'success': True}


@app.route('/debug', methods=['POST'])
def handle_debug_request():
    sample_count = int(request.form.get('sample_count'))
    only_log_errors = request.form.get('only_log_errors') == 'true' 
    keep_existing_errors = request.form.get('keep_existing_errors') == 'true' 
    print(only_log_errors, keep_existing_errors)
    Logger.init_logger(only_log_errors, keep_existing_errors)
    run_debugger(num_samples=sample_count)
    generate_html_file()
    return {'success': True}


def run_frontend():
    app.run(debug=True)
