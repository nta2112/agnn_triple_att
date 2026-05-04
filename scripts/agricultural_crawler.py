import os
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

def crawl_agricultural_images(base_dir, regions, products, max_num=50):
    """
    Crawls images for each region-product combination.
    
    Args:
        base_dir (str): The root directory to save images.
        regions (list): List of regions/countries.
        products (list): List of agricultural products.
        max_num (int): Maximum number of images to download per category.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for region in regions:
        for product in products:
            # Create folder name following the pattern in your example
            # We use '_' but you can change it to '-' if preferred
            folder_name = f"{region.lower()}_{product.lower()}"
            save_path = os.path.join(base_dir, folder_name)
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            print(f"--- Crawling for: {region} {product} ---")
            
            # Construct a clear search query
            # Adding 'fruit' or 'vegetable' helps get better results
            query = f"{region} {product} fruit" if product.lower() in ['apple', 'grape', 'orange', 'strawberry'] else f"{region} {product} vegetable"
            
            # Using BingImageCrawler as it's often more permissive than Google
            # You can also use GoogleImageCrawler if you have the setup
            bing_crawler = BingImageCrawler(downloader_threads=4, storage={'root_dir': save_path})
            bing_crawler.crawl(keyword=query, max_num=max_num)

if __name__ == "__main__":
    # Define the output path based on your request
    # You might want to change this to a new folder like 'crawled_data'
    OUTPUT_PATH = r"C:\Users\HP\OneDrive\Desktop\AGNN_01\AGNN\data\crawled_images"
    
    # Categories extracted from your example
    REGIONS = [
        "Vietnam", "America", "China", "Dalat", "Japan", 
        "Korea", "Australia", "Netherlands", "New Zealand", "South Africa"
    ]
    
    PRODUCTS = ["Apple", "Grape", "Orange", "Potato", "Strawberry"]
    
    # Run the crawler
    # Start with a small max_num to test (e.g., 5 or 10)
    crawl_agricultural_images(OUTPUT_PATH, REGIONS, PRODUCTS, max_num=20)
