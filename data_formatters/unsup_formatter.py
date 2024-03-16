import csv
import hashlib

def generate_id(sentence):
    # Hash the sentence to create a unique ID
    # Adjust the [:16] part to change the length of the ID
    return hashlib.sha256(sentence.encode()).hexdigest()[:16]

def txt_to_csv(txt_file_path):
    # Open the text file and read the content
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Open a new CSV file to write the sentences into
    with open('simcse/unsup_simcse.csv', 'w', newline='', encoding='utf-8') as csvfile:
        # Create a csv writer object
        csvwriter = csv.writer(csvfile, delimiter='\t')
        
        # Label the columns in the first line
        csvwriter.writerow(['id', 'sentence'])
        
        # Write each line to the csv, with a generated id
        for line in lines:
            # Clean the line to remove newlines
            clean_line = line.strip()
            
            # Generate a unique ID for each sentence
            row_id = generate_id(clean_line)
            
            csvwriter.writerow([row_id, clean_line])

txt_to_csv('simcse/wiki1m_for_simcse.txt')
