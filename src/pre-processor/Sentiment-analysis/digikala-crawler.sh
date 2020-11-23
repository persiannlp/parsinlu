#! /bin/bash
# This script is used to crawl reviews from Digikala
git clone https://github.com/rajabzz/digikala-crawler.git
cd digikala-crawler
pip install -r requirements.txt
scrapy crawl comment -o digikala-results.jl 
echo "Done!"
