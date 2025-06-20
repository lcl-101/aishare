'''
docker 部署命令
docker run -d -p 11235:11235 --name crawl4ai --shm-size=1g unclecode/crawl4ai:latest
docker 删除命令
docker stop crawl4ai
docker rm crawl4ai
docker image rm unclecode/crawl4ai

'''
'''
SDK 部署
mkdir crawl4ai
cd crawl4ai
python -m venv .
.\Scripts\activate
pip install crawl4ai
'''

import asyncio
from crawl4ai import AsyncWebCrawler

async def main():
    # Create an instance of AsyncWebCrawler
    async with AsyncWebCrawler() as crawler:
        # Run the crawler on a URL
        result = await crawler.arun(url="https://crawl4ai.com")

        # Print the extracted content
        print(result.markdown)

# Run the async main function
asyncio.run(main())
