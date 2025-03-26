import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def get_sample_questions():
    """
    Get a list of sample questions.

    Returns:
    list: A list of sample questions.
    """
    return [
        "My laptop is running very slow. How can I fix it?",
        "WiFi keeps disconnecting frequently. What should I do?",
        "I forgot my system password. How can I reset it?",
        "My screen is flickering. How can I troubleshoot it?",
        "How do I fix a 'Blue Screen of Death' error?",
        "I lost access to my corporate email. How can I recover it?",
        "I’m unable to log into my VPN. What should I check?",
        "My antivirus keeps blocking a trusted application. How do I allow it?",
        "I got a phishing email. What should I do?",
        "I need admin access to install software. How can I request it?",
        "Outlook is not syncing emails. How can I fix this?",
        "Teams is not working on my laptop. How can I troubleshoot?",
        "I’m getting a 'License Expired' error in Microsoft Office. How do I fix it?",
        "I accidentally deleted an important email. How can I recover it?",
        "My email attachments are not downloading. What should I do?",
        "My printer is not working. How can I fix it?",
        "External USB drive is not getting detected. How do I fix it?",
        "My keyboard is typing the wrong characters. What should I do?",
        "The second monitor is not being detected. How can I troubleshoot?",
        "Headphones are not working on my laptop. How can I fix this?"
    ]