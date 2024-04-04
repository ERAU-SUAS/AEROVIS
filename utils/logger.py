import os
import shutil
import re
import webbrowser

HTML_FILE = 'log/index.html'
LOG_DIR = "log"
LOG_RESULT_IMG_DIR = f"{LOG_DIR}/pics"
LOG_FILE_NAME = "log"

def toggle_error(target_line_no, is_checked):
    content = ""
    with open(LOG_DIR + f"/{LOG_FILE_NAME}.csv", "r") as f: 
        for line_no, line in enumerate(f.readlines()):
            if line_no == target_line_no:
                spt = line.split(',')
                if (spt[0] == "ERROR" and is_checked) or (spt[0] != "ERROR" and not is_checked):
                    content += line
                elif spt[0] == "ERROR":
                    content += ''.join(entry + ',' for entry in spt[2:])[:-1]
                elif spt[0] != "ERROR" and is_checked:
                    content += "ERROR,flagged as error," + line 
            else:
                content += line
    with open(LOG_DIR + f"/{LOG_FILE_NAME}.csv", "w") as f: 
        f.write(content)


def keep_errors():
    stuff_to_keep = ""
    imgs_to_keep = []
    with open(LOG_DIR + f"/{LOG_FILE_NAME}.csv", "r") as f: 
        for line in f.readlines():
            spt = line.split(',')
            if spt[0] == "ERROR":
                match = re.search(r'pics/([a-z0-9]+)_', spt[-1])
                imgs_to_keep.append(match.group(1))
                stuff_to_keep += line
    with open(LOG_DIR + f"/{LOG_FILE_NAME}.csv", "w") as f: 
        f.write(stuff_to_keep) 

    for filename in os.listdir(LOG_RESULT_IMG_DIR):
        thing = filename.split('_')[0] 
        if thing not in imgs_to_keep:
            file_path = os.path.join(LOG_RESULT_IMG_DIR, filename)
            os.remove(file_path)


class Logger():
    only_errors_flag = False
    log_file = None 

    def init_logger(only_errors_flag, keep_errors_flag):
        Logger.only_errors_flag = only_errors_flag 
        if keep_errors_flag: 
            try:
                os.remove(HTML_FILE)
            except OSError:
                pass
            keep_errors()
        else:
            shutil.rmtree(LOG_DIR, ignore_errors=True)

        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(LOG_RESULT_IMG_DIR, exist_ok=True)
        Logger.log_file = open(LOG_DIR + f"/{LOG_FILE_NAME}.csv", "a") 

    def log(character, shape_color, character_color, last_image_path, og_img_path, error_msg=None): 
        error = ""
        if error_msg is None and Logger.only_errors_flag:
            return
        elif error_msg is not None:
            error = f"ERROR,{error_msg},"

        # inb4 "why not just pass an array" - does this not look like more fun?!!?/!11!1!!??/!!
        match = re.search(r'(\d+)(?=\.[^\.]+$)', last_image_path) 
        num_imgs = int(match.group(1)) # impossible for no matches. whats an error? 
        img_paths = ""

        for this_is_a_really_long_name_for_a_variable_lol_especially_when_i_couldve_just_used_i in range(num_imgs): 
            number = this_is_a_really_long_name_for_a_variable_lol_especially_when_i_couldve_just_used_i + 1
            img_paths += re.sub(r'(_\d+)(\.\w+)$', r'_{}\2'.format(number), last_image_path) + ","

        Logger.log_file.write(f"{error}{character},{shape_color},{character_color},{og_img_path},{img_paths[:-1]}\n")

    def close_log_file():
        Logger.log_file.close()

# thanks gpt :D
def generate_html_file():
    # Assuming your CSV file is named 'data.csv' and located in the same directory as your script
    csv_file_path = 'log/log.csv'  # Update this to your CSV file path

    # Use 'with' statement to open and read the file
    Logger.close_log_file()
    with open(csv_file_path, 'r') as file:
        csv_data = file.read()

    # Now you can use the method to split the content into lines
    lines = csv_data.strip().split('\n')

    # Add table rows for each line in the CSV data
    num_imgs = 0
    imgs_html_content = ""
    for counter, line in enumerate(lines):
        splt = line.split(',') 
        error = splt[0] == "ERROR"
        error_offset = 0
        tr_text = "<tr>"
        er_text = ""

        if error: 
            error_offset = 2
            tr_text = '<tr color="#F00">'
            er_text = f"<td>{splt[1]}</td>"

        character, color1, color2 = splt[error_offset:3 + error_offset] 
        img_paths = splt[3 + error_offset:]
        num_imgs = max(num_imgs, len(img_paths) - 1)
        img_path_content = ""
        for img_path in img_paths:
            img_path_content += f'\t\t\t\t<td><img src="{img_path[4:]}" alt="Image" height="100"></td>\n'
        checked_text = "checked" if error else ""
        imgs_html_content += f"""
            {tr_text}
                <td><input type="checkbox" id="cb{counter + 1}" onchange="toggleError({counter + 1}, this.checked)" {checked_text} autocomplete="off"></td>
                <td>{counter + 1}</td>
                <td>{character}</td>
                <td>{color1}</td>
                <td>{color2}</td>
{img_path_content}
                {er_text}
            </tr>
        """

    # wtf is python this is fun lol
    headers_for_imgs = ''.join([f"<th>Result {index+1}</th>\n\t\t\t\t" for index in range(num_imgs)])

    # Prepare the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CSV Data Table</title>
        <style>
            table, th, td {{
                border: 1px solid black;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 10px;
                text-align: left;
            }}
        </style>
    </head>
    <body>
        <span>Sample Count: <input type="text" value="20" id="sampleCount"></input></span>
        <span>Only Log Errors: <input type="checkbox" id="onlyLogErrors"></input></span>
        <span>Keep Existing Errors: <input type="checkbox" id="keepExistingErrors"></input></span>
        <button onClick="debug()" id="debugButton">Run Script</button>

        <table>
            <tr>
                <th>Flag Error</th>
                <th>#</th>
                <th>Character</th>
                <th>Shape Color</th>
                <th>Character Color</th>
                <th>Original Image</th>
                {headers_for_imgs}
            </tr>
{imgs_html_content}
    """

    # Close the table and the HTML tags
    html_content += """
        </table>

        <script>
            function debug() {
                document.getElementById("debugButton").disabled = true; 

                sc = document.getElementById("sampleCount").value;
                ole = document.getElementById("onlyLogErrors").checked;
                kee = document.getElementById("keepExistingErrors").checked;
            
                fetch('/debug', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `sample_count=${sc}&only_log_errors=${ole}&keep_existing_errors=${kee}`,
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById("debugButton").disabled = false; 
                        window.location.reload();
                    } else {
                        alert("Error running debug script");
                    }
                })
                .catch(error => console.error('Error:', error));
            }

            function toggleError(customIdentifier, isChecked) {
                fetch('/toggle-error', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `identifier=${customIdentifier}&is_checked=${isChecked}`,
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        window.location.reload();
                    } else {
                        alert("Error updating error status");
                    }
                })
                .catch(error => console.error('Error:', error));
            }
        </script>

    </body>
    </html>
    """

    # Write the HTML content to a file
    with open(HTML_FILE, 'w') as html_file:
        html_file.write(html_content)


def open_log():
    new = 2 # open in a new tab, if possible
    path = os.path.abspath(HTML_FILE)
    url = f"file://{path}"
    webbrowser.open(url,new=new)
