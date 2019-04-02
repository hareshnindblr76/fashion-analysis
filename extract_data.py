import json,os
import requests
annotation_file = 'product_data.json'
output_dir = 'fashion_data'
categories_file = 'product_categories.txt'
def main():
    product_data = json.load(open(annotation_file))
    product_categories = open(categories_file).read().split('\n')
    print(product_categories)
    output_json = []
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for product in product_data:
        url = product["images_url"]
        img_name = url.split('/')[-1]
        img_name = img_name[:img_name.find('?')]
        if not img_name.endswith('.jpg'):
            img_name+='.jpg'
        description = product["description"]
        category = ''
        loc = description.find('Free Shipping')
        if loc!=-1:
            category = description[:loc-1].split()[-1][:-1]
            print(category)
            if category not in product_categories:
                category = 'Others'
        if category!='':
            output_json.append({"image_path":img_name, "class":product_categories.index(category)})

        if not os.path.exists(os.path.join(output_dir, img_name)):
            print("Downloading image ",img_name)
            if not url.startswith('http'):
                url = "https:"+url

            try:
                with open(os.path.join(output_dir, img_name),'wb') as handler:
                    handler.write(requests.get(url).content)
            except:
                pass
    json.dump(output_json,open("annotations.json","w"),indent=4)

if __name__=='__main__':
    main()