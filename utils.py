import os
from werkzeug.utils import secure_filename

# Function to preprocess the answer (replace underscores with spaces)
def preprocess_answer(answer):
    """
    Cleans up the predicted answer by replacing underscores with spaces.
    """
    return answer.replace("_", " ")

# Function to save the uploaded file securely
def save_uploaded_file(file, upload_folder):
    """
    Saves the uploaded file to the specified folder securely.
    
    Args:
        file: The uploaded file object.
        upload_folder: Path to the folder where the file will be saved.

    Returns:
        str: The full path to the saved file.
    """
    # Ensure the upload folder exists
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Securely save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)
    return file_path
