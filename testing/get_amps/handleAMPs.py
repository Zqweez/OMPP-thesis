"""
Sort the CSV file in ID order and separate all helices into a different file.
"""
import csv
import os

def sort_amps(file_name):
    """Sort AMPs based on their ID and extract helices"""
    amps = []
    helices = []

    with open(file_name, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            amps.append(row)
            if row['Structure_3D'] == 'Helix':
                helices.append(row)

    # Sort by ID
    amps.sort(key=lambda x: int(x['ID'][2:]))
    helices.sort(key=lambda x: int(x['ID'][2:]))

    # Write sorted AMPs to new CSV
    sorted_file_name = file_name.replace('.csv', '_sorted.csv')
    with open(sorted_file_name, 'w', newline='') as csvfile:
        fieldnames = amps[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(amps)

    # Write helices to separate CSV
    helices_file_name = file_name.replace('.csv', '_helices.csv')
    with open(helices_file_name, 'w', newline='') as csvfile:
        fieldnames = helices[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(helices)

    print(f"Sorted AMPs saved to {sorted_file_name}")
    print(f"Helices saved to {helices_file_name}")

if __name__ == "__main__":
    source_file = os.path.dirname(__file__)
    file_name = os.path.join(source_file, f'amp_data_1-6338.csv')
    sort_amps(file_name)