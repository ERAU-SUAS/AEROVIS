import os
import shutil
import re
import webbrowser

HTML_FILE = 'log/index.html'
LOG_DIR = "log"
LOG_RESULT_IMG_DIR = f"{LOG_DIR}/pics"
LOG_FILE_NAME = "log"


class Logger():
    shutil.rmtree(LOG_DIR, ignore_errors=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(LOG_RESULT_IMG_DIR, exist_ok=True)
    log_file = open(LOG_DIR + f"/{LOG_FILE_NAME}.csv", "a") 

    def log(character, shape_color, character_color, last_image_path, og_img_path, error_msg=None): 
        # inb4 "why not just pass an array" - does this not look like more fun?!!?/!11!1!!??/!!
        match = re.search(r'(\d+)(?=\.[^\.]+$)', last_image_path) 
        num_imgs = int(match.group(1)) # impossible for no matches. whats an error? 
        img_paths = ""

        for this_is_a_really_long_name_for_a_variable_lol_especially_when_i_couldve_just_used_i in range(num_imgs): 
            number = this_is_a_really_long_name_for_a_variable_lol_especially_when_i_couldve_just_used_i + 1
            img_paths += re.sub(r'(_\d+)(\.\w+)$', r'_{}\2'.format(number), last_image_path) + ","

        error = ""
        if error_msg is not None:
            error = f"ERROR,{error_msg},"

        Logger.log_file.write(f"{error}{character},{shape_color},{character_color},{og_img_path},{img_paths[:-1]}\n")

    def close_log_file():
        Logger.log_file.close()


# thanks gpt4 :D
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
        num_imgs = len(img_paths) - 1
        img_path_content = ""
        for img_path in img_paths:
            img_path_content += f'\t\t\t\t<td><img src="{img_path[4:]}" alt="Image" height="100"></td>\n'
        imgs_html_content += f"""
            {tr_text}
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
        <table>
            <tr>
                <th>#</th>
                <th>Character</th>
                <th>Shape Color</th>
                <th>Character Color</th>
                <th>Original Image</th>
                {headers_for_imgs}
                <th>Error Info</th>
            </tr>
{imgs_html_content}
    """

    # Close the table and the HTML tags
    html_content += """
        </table>
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
