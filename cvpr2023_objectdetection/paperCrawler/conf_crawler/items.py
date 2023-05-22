# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from scrapy import Field, Item


class Paper(Item):

    conference = Field()
    year = Field()
    title = Field()
    authors = Field()
    abstract = Field()
    keywords = Field()
    code_url = Field()
    pdf_url = Field()
    matched_queries = Field()
