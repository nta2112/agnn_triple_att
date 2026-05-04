import os
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler

def crawl_images_by_keywords(base_dir, keywords, max_num=20):
    """
    Crawls images for a list of keywords and organizes them into folders.
    
    Args:
        base_dir (str): Root directory to save data.
        keywords (list): List of labels/keywords (e.g., 'chanh mỹ', 'khoai tây đà lạt').
        max_num (int): Number of images to download per label.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for keyword in keywords:
        # Folder name is exactly the keyword label
        folder_name = keyword.strip()
        save_path = os.path.join(base_dir, folder_name)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        print(f"\n=== Đang cào dữ liệu cho nhãn: {folder_name} ===")
        
        # We use BingImageCrawler as it's very reliable
        # You can add GoogleImageCrawler if needed
        crawler = BingImageCrawler(downloader_threads=4, storage={'root_dir': save_path})
        
        # Search query is the keyword
        crawler.crawl(keyword=keyword, max_num=max_num)
        print(f"Hoàn thành tải {max_num} ảnh cho nhãn '{folder_name}' vào thư mục: {save_path}")

if __name__ == "__main__":
    # Đường dẫn lưu dữ liệu (theo yêu cầu của bạn)
    OUTPUT_DIR = r"C:\Users\HP\OneDrive\Desktop\AGNN_01\AGNN\data\crawled_data"
    
    # Danh sách các từ khóa (nhãn) của bạn
    # Bạn có thể thêm bất kỳ từ khóa nào vào đây
    MY_KEYWORDS = [
        "chanh mỹ",
        "khoai tây đà lạt",
        "táo nhật",
        "nho ninh thuận"
    ]
    
    # Số lượng ảnh tối đa cho mỗi nhãn
    MAX_IMAGES_PER_LABEL = 30
    
    crawl_images_by_keywords(OUTPUT_DIR, MY_KEYWORDS, max_num=MAX_IMAGES_PER_LABEL)
