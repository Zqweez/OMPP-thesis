"""
Get the data for all AMPs from APD database. 
Read the raw html response and save the information about each AMP as a row in a CSV file.
Next step open the article that discovered the AMP and extract more information about the AMP.
Filter the article for keywords such as "NPN", "Disc35"

ID between 00001 and 06338

Example cURL request:
curl 'https://aps.unmc.edu/database/peptide' \
  -H 'Connection: keep-alive' \
  -H 'Origin: https://aps.unmc.edu' \
  -H 'Referer: https://aps.unmc.edu/database/result' \
  --data-raw 'ID=00001'
"""

import os
import requests
import csv
import time
import re
from bs4 import BeautifulSoup
import urllib3
from colorama import Fore, Style
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_amp_data(amp_id):
    """Fetch AMP data for a specific ID"""
    url = "https://aps.unmc.edu/database/peptide"
    headers = {
        'Connection': 'keep-alive',
        'Origin': 'https://aps.unmc.edu',
        'Referer': 'https://aps.unmc.edu/database/result',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    data = {'ID': f"{amp_id:05d}"}
    
    try:
        response = requests.post(url, headers=headers, data=data, timeout=30, verify=False)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching AMP ID {amp_id:05d}: {e}")
        return None

def parse_amp_html(html_content, amp_id):
    """Parse HTML content to extract AMP information"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Check if the page contains valid AMP data
    peptide_title = soup.find('h1', class_='peptide')
    if not peptide_title or 'AP' not in peptide_title.get_text():
        print(f"No valid AMP data found for ID {amp_id:05d}")
        return None
    
    # Find the main table containing peptide information
    table = soup.find('table', class_='peptide')
    if not table:
        print(f"No table found for AMP ID {amp_id:05d}")
        return None
    
    amp_data = {'ID': f"AP{amp_id:05d}"}
    
    # Parse table rows
    rows = table.find_all('tr')
    for row in rows:
        cells = row.find_all('td')
        if len(cells) >= 2:
            field_name = cells[0].get_text(strip=True).replace(':', '')
            field_value = cells[1].get_text(strip=True)
            
            # Clean up field values
            field_value = re.sub(r'\s+', ' ', field_value)  # Remove extra whitespace
            
            # Map fields to standardized names
            field_mapping = {
                'Name/Class': 'Name_Class',
                'Source': 'Source',
                'Sequence': 'Sequence',
                'Length': 'Length',
                'Net charge': 'Net_Charge',
                'Hydrophobic residue%': 'Hydrophobic_Percentage',
                'Boman Index': 'Boman_Index',
                '3D Structure': 'Structure_3D',
                'Method': 'Method',
                'SwissProt ID': 'SwissProt_ID',
                'Activity': 'Activity',
                'Crucial residues': 'Crucial_Residues',
                'Additional info': 'Additional_Info',
                'Title': 'Title',
                'Author': 'Author',
                'Reference': 'Reference'
            }
            
            if field_name in field_mapping:
                amp_data[field_mapping[field_name]] = field_value
            if field_name == 'Reference':
                # Extract PubMed URL if available
                pubmed_link = cells[1].find('a', href=True)
                if pubmed_link:
                    amp_data['PubMed_URL'] = pubmed_link['href']
        
    return amp_data

def save_to_csv(amp_data_list, filename='amp_database.csv'):
    """Save AMP data to CSV file"""
    if not amp_data_list:
        print("No data to save")
        return
    
    # Define CSV columns
    fieldnames = ['ID', 'APD_ID', 'Name_Class', 'Source', 'Sequence', 'Length', 
                 'Net_Charge', 'Hydrophobic_Percentage', 'Boman_Index', 'Structure_3D',
                 'Method', 'SwissProt_ID', 'Activity', 'Crucial_Residues', 
                 'Additional_Info', 'Title', 'Author', 'Reference', 'PubMed_URL']
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for data in amp_data_list:
            # Ensure all fields exist with empty string as default
            row = {field: data.get(field, '') for field in fieldnames}
            writer.writerow(row)
    
    print(f"Data saved to {filename} with {len(amp_data_list)} records")

def scrape_amp_range(start_id=1, end_id=5, delay=1):
    """Scrape AMP data for a range of IDs"""
    amp_data_list = []
    total_ids = end_id - start_id + 1
    
    for i, amp_id in enumerate(range(start_id, end_id + 1)):
        # Calculate progress
        progress = (i + 1) / total_ids
        percentage = int(progress * 100)
        
        # Create simple progress bar
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Print progress bar
        print(f"\r[{bar}] {percentage:3d}% ({i + 1}/{total_ids}) - Fetching ID: {amp_id:05d}", end='', flush=True)
        
        html_content = fetch_amp_data(amp_id)
        if html_content:
            amp_data = parse_amp_html(html_content, amp_id)
            if amp_data:
                amp_data_list.append(amp_data)
            else:
                print(f"\n{Fore.RED}✗ Failed to parse AMP ID: {amp_id:05d}{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}✗ Failed to fetch AMP ID: {amp_id:05d}{Style.RESET_ALL}")

        # Be respectful to the server
        if amp_id < end_id:  # Don't sleep after the last request
            time.sleep(delay)
    
    print()  # New line after progress bar completes
    return amp_data_list

def fetch_single_amp(amp_id, delay=0.5):
    """Fetch a single AMP with rate limiting for threading"""
    time.sleep(delay)  # Rate limiting per thread
    html_content = fetch_amp_data(amp_id)
    if html_content:
        amp_data = parse_amp_html(html_content, amp_id)
        return amp_data
    return None

def scrape_amp_range_threaded(start_id=1, end_id=5, max_workers=3, delay=0.5):
    """Scrape AMP data using multiple threads with rate limiting"""
    amp_data_list = []
    total_ids = end_id - start_id + 1
    completed = 0
    lock = threading.Lock()
    
    def update_progress():
        nonlocal completed
        with lock:
            completed += 1
            progress = completed / total_ids
            percentage = int(progress * 100)
            
            # Create simple progress bar
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            print(f"\r[{bar}] {percentage:3d}% ({completed}/{total_ids}) - Processing...", end='', flush=True)
    
    print(f"Using {max_workers} threads with {delay}s delay per request...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_id = {
            executor.submit(fetch_single_amp, amp_id, delay): amp_id 
            for amp_id in range(start_id, end_id + 1)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_id):
            amp_id = future_to_id[future]
            try:
                result = future.result()
                if result:
                    with lock:
                        amp_data_list.append(result)
                else:
                    print(f"\n{Fore.RED}✗ Failed to process AMP ID: {amp_id:05d}{Style.RESET_ALL}")
            except Exception as e:
                print(f"\n{Fore.RED}✗ Error processing AMP ID {amp_id:05d}: {e}{Style.RESET_ALL}")
            
            update_progress()
    
    print()  # New line after progress bar completes
    return amp_data_list

def main():
    """Main function to run the AMP scraper"""
    start_id = 4836
    end_id = 4956
    
    # Configuration - change these settings
    USE_THREADING = True  # Set to False for single-threaded
    MAX_WORKERS = 3       # Number of concurrent threads (keep low!)
    THREAD_DELAY = 0.5    # Delay per thread (seconds)
    SINGLE_DELAY = 1      # Delay for single-threaded (seconds)
    
    print(Fore.YELLOW + f"Starting AMP scraping for IDs {start_id}-{end_id}" + Style.RESET_ALL)
    total_requested = end_id - start_id + 1
    
    # Choose scraping method
    if USE_THREADING:
        print(Fore.BLUE + f"Using multithreading: {MAX_WORKERS} workers, {THREAD_DELAY}s delay each" + Style.RESET_ALL)
        amp_data = scrape_amp_range_threaded(start_id=start_id, end_id=end_id, 
                                           max_workers=MAX_WORKERS, delay=THREAD_DELAY)
    else:
        print(Fore.BLUE + f"Using single-threaded: {SINGLE_DELAY}s delay between requests" + Style.RESET_ALL)
        amp_data = scrape_amp_range(start_id=start_id, end_id=end_id, delay=SINGLE_DELAY)

    if amp_data:
        # Get location of source file for saving CSV in the same directory
        source_file = __file__
        source_dir = os.path.dirname(source_file)
        method = "threaded" if USE_THREADING else "single"
        csv_file = os.path.join(source_dir, f'amp_data_{start_id}-{end_id}_{method}.csv')

        save_to_csv(amp_data, csv_file)

        # Print summary with success rate
        success_rate = (len(amp_data) / total_requested) * 100
        print(Fore.GREEN + f"✓ Scraping completed! " + Style.RESET_ALL)
        print(Fore.CYAN + f"  Collected: {len(amp_data)}/{total_requested} records ({success_rate:.1f}% success rate)" + Style.RESET_ALL)
        print(Fore.CYAN + f"  Saved to: {csv_file}" + Style.RESET_ALL)
    else:
        print(Fore.RED + "No data collected" + Style.RESET_ALL)

if __name__ == "__main__":
    main()