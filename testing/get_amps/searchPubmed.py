""" Outdated - testing script
Multi-threaded PubMed keyword search with automatic backoff and error threshold.
Searches for specified keywords in PubMed publications, urls extracted from a CSV file.
Saves URLs with found keywords to an output CSV file.
"""
import requests
import csv
import time
import os
from colorama import Fore, Style
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3
import logging
from queue import Queue
import random

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging to avoid interference with progress bar
logging.basicConfig(level=logging.WARNING)  # Only show warnings and errors during execution

def get_pubmed_urls(file_path):
    """Extract PubMed URLs from the AMP data CSV file."""
    pubmed_urls = []
    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if 'PubMed_URL' in row and row['PubMed_URL']:
                pubmed_urls.append(row['PubMed_URL'])
    return pubmed_urls

def search_keywords_in_pubmed(url, keywords, found_messages_queue=None, max_retries=3):
    """Search for keywords in the PubMed publication with automatic backoff for rate limiting."""
    backoff_codes = {403, 429, 503, 502, 504}  # HTTP codes that should trigger backoff
    
    for attempt in range(max_retries + 1):
        try:
            # Add some jitter to avoid thundering herd
            if attempt > 0:
                backoff_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(backoff_time)
                logging.warning(f"Retrying {url} (attempt {attempt + 1}) after {backoff_time:.1f}s backoff")
            
            response = requests.get(url, verify=False, timeout=15)
            
            if response.status_code == 200:
                keywords_dict = {keyword: False for keyword in keywords}
                content = response.text.lower()
                found_keywords = []
                
                for keyword in keywords:
                    if keyword.lower() in content:
                        keywords_dict[keyword] = True
                        found_keywords.append(keyword)
                        
                # Queue the messages instead of printing immediately
                if found_keywords and found_messages_queue:
                    for keyword in found_keywords:
                        found_messages_queue.put(f"Keyword '{keyword}' found in {url}")
                        
                return (url, keywords_dict) if any(keywords_dict.values()) else None
                
            elif response.status_code in backoff_codes:
                if attempt < max_retries:
                    logging.warning(f"Rate limited (HTTP {response.status_code}) for {url}, backing off...")
                    continue
                else:
                    logging.error(f"Max retries exceeded for {url} (HTTP {response.status_code})")
                    return None
            else:
                logging.warning(f"HTTP {response.status_code} for {url}")
                return None
                
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout for {url} (attempt {attempt + 1})")
            if attempt == max_retries:
                return None
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
            if attempt == max_retries:
                return None
    
    return None

def threaded_search(pubmed_urls, keywords, max_workers=5, delay=0.5, max_errors=10):
    """Perform multi-threaded search for keywords in PubMed URLs.
    Stops early if max_errors threshold is exceeded."""
    completed = 0
    found_urls = []
    total_urls = len(pubmed_urls)
    lock = threading.Lock()
    found_messages_queue = Queue()
    error_count = 0
    early_stop = False

    def update_progress():
        nonlocal completed
        with lock:
            completed += 1
            progress = completed / total_urls
            percentage = int(progress * 100)
            
            # Create simple progress bar
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Clear line and print progress
            print(f"\r[{bar}] {percentage:3d}% ({completed}/{total_urls}) - Processing...", end='', flush=True)

    print(f"Starting search across {total_urls} PubMed URLs...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(search_keywords_in_pubmed, url, keywords, found_messages_queue): url for url in pubmed_urls}
        
        for future in as_completed(future_to_url):
            # Check if we should stop early due to too many errors
            with lock:
                if error_count >= max_errors:
                    early_stop = True
                    print(f"\n{Fore.RED}Stopping search early: {error_count} errors exceeded threshold of {max_errors}{Style.RESET_ALL}")
                    # Cancel remaining futures
                    for remaining_future in future_to_url.keys():
                        if not remaining_future.done():
                            remaining_future.cancel()
                    break
            
            url = future_to_url[future]
            try:
                result = future.result()
                if result:
                    found_urls.append(result)
            except Exception as e:
                with lock:
                    error_count += 1
                logging.error(f"Error processing {url}: {e}")
            
            # Adaptive delay based on error rate
            current_error_rate = error_count / max(1, completed)
            adaptive_delay = delay * (1 + current_error_rate * 2)  # Increase delay if many errors
            time.sleep(adaptive_delay)
            
            update_progress()
            
    print()  # New line after progress bar
    
    # Print all the found keyword messages after progress bar is done
    if not found_messages_queue.empty():
        print(Fore.CYAN + "\n=== Keyword Search Results ===" + Style.RESET_ALL)
        while not found_messages_queue.empty():
            message = found_messages_queue.get()
            print(Fore.YELLOW + message + Style.RESET_ALL)
    else:
        print(Fore.YELLOW + "\nNo keyword matches found in any publications." + Style.RESET_ALL)
    
    if early_stop:
        print(Fore.RED + f"\nSearch stopped early after {error_count} errors. Processed {completed}/{total_urls} URLs." + Style.RESET_ALL)
        print(Fore.YELLOW + "Saving partial results..." + Style.RESET_ALL)
    elif error_count > 0:
        print(Fore.RED + f"\nEncountered {error_count} errors during processing." + Style.RESET_ALL)
    
    return found_urls

if __name__ == "__main__":
    source_file = os.path.dirname(__file__)
    file_name = os.path.join(source_file, f'amp_data_1-500_threaded.csv')
    # Get PubMed URLs from the specified CSV file
    pubmed_urls = get_pubmed_urls(file_name)

    print(Fore.CYAN + f"Found {len(pubmed_urls)} PubMed URLs to process" + Style.RESET_ALL)

    # Search each publication for keywords
    keywords = ['NPN', 'OMPP', 'Disc35', 'bilayer', 'outer membrane', 'LPS', 'lipopolysaccharide']
    found_urls = [] 
    # Multi-threaded search with error threshold
    found_urls = threaded_search(pubmed_urls, keywords, max_workers=2, delay=1.0, max_errors=10)

    if found_urls:
        print(Fore.GREEN + f"Search completed. Keywords found in {len(found_urls)} publications." + Style.RESET_ALL)
    else:
        print(Fore.YELLOW + "No publications with matching keywords were found." + Style.RESET_ALL)

    # Save found URLs to a CSV file
    output_file = os.path.join(source_file, f'found_pubmed_urls.csv')
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['PubMed_URL'])
        for url in found_urls:
            writer.writerow([url])
    print(Fore.GREEN + f"Found URLs saved to {output_file}" + Style.RESET_ALL)
