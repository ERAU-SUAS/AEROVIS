# thanks gpt4 :D

HTML_FILE = 'log/index.html'

def generate_html_file():
    # Assuming your CSV file is named 'data.csv' and located in the same directory as your script
    csv_file_path = 'log/log.csv'  # Update this to your CSV file path

    # Use 'with' statement to open and read the file
    with open(csv_file_path, 'r') as file:
        csv_data = file.read()

    # Now you can use the method to split the content into lines
    lines = csv_data.strip().split('\n')

    # Prepare the HTML content
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CSV Data Table</title>
        <style>
            table, th, td {
                border: 1px solid black;
                border-collapse: collapse;
            }
            th, td {
                padding: 10px;
                text-align: left;
            }
        </style>
    </head>
    <body>
        <table>
            <tr>
                <th>Character</th>
                <th>Shape Color</th>
                <th>Character Color</th>
                <th>Original Image</th>
                <th>Test Result</th>
            </tr>
    """

    # Add table rows for each line in the CSV data
    for line in lines:
        id, color1, color2, image_path, og_image_path = line.split(',')
        html_content += f"""
            <tr>
                <td>{id}</td>
                <td>{color1}</td>
                <td>{color2}</td>
                <td><img src="{og_image_path[4:]}" alt="Image" height="100"></td>
                <td><img src="{image_path[4:]}" alt="Image" height="100"></td>
            </tr>
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
