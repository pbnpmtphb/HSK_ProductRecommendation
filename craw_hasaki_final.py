import requests
from bs4 import BeautifulSoup
import pandas as pd

# Base URL with placeholder for page number
base_url = 'https://hasaki.vn/danh-muc/cham-soc-da-mat-c4.html?p={}'

# Lists to store data from all pages
product_names_list = []
product_images_list = []
product_links_list = []
product_ids_list = []

# Loop through pages 1 to 68
for page in range(1, 69):
    print(f"Scraping page {page}...")
    # Construct page URL
    page_link = base_url.format(page)
    
    # Get the page content
    page_response = requests.get(page_link)
    page_content = BeautifulSoup(markup=page_response.content, features='html.parser')
    
    # Find all product containers
    div_product = page_content.find_all(name='div', attrs={'class': 'ProductGridItem__itemOuter'})
    
    # Extract data from each product on the current page
    for product in div_product:
        # Extract product name
        product_name = product.find(name='div', attrs={'class': 'vn_names'})
        product_names_list.append(product_name.text.strip() if product_name else None)
        
        # Extract product image URL
        product_image = product.find(name='img', attrs={'class': 'photo image img_thumb_sub loading'})
        product_images_list.append(product_image['src'] if product_image else None)
        
        # Extract product link
        product_link = product.find(name='a', href=True)
        product_links_list.append(product_link['href'] if product_link else None)

# Extract product IDs from product URLs
print("Extracting product IDs...")
for product_url in product_links_list:
    if product_url:
        try:
            # Send request to the product page
            product_response = requests.get(product_url)
            product_content = BeautifulSoup(markup=product_response.content, features='html.parser')
            
            # Find the SKU span
            sku_span = product_content.find(name='span', attrs={'class': 'item-sku txt_color_1'})
            
            if sku_span:
                # Extract the SKU number
                sku_text = sku_span.text.strip()
                sku_number = sku_text.split(":")[-1].strip()
                product_ids_list.append(sku_number)
            else:
                product_ids_list.append(None)
        except Exception as e:
            print(f"Error extracting product ID for URL {product_url}: {e}")
            product_ids_list.append(None)
    else:
        product_ids_list.append(None)

# Create a DataFrame from the collected data
df_products = pd.DataFrame({
    'name': product_names_list,
    'image': product_images_list,
    'url': product_links_list,
    'productid': product_ids_list
})

# Save the DataFrame to a CSV file
csv_file_path = 'products_with_ids.csv'
df_products.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

print(f"Data from all pages has been successfully saved to {csv_file_path}")