import inspect
import re
from abc import abstractmethod

import scrapy

from ..items import Paper


class BaseSpider(scrapy.Spider):

    def __init__(self, years, queries=None, *args, **kwargs):
        super(BaseSpider, self).__init__(*args, **kwargs)

        self.years = [int(y) for y in years.split(',')]
        self.queries = queries.split(',') if queries is not None else None

    @abstractmethod
    def parse_paper_details(self, response, year):
        pass


class CVFSpider(BaseSpider):

    def start_requests(self):
        base_url = 'https://openaccess.thecvf.com/'
        for year in self.years:
            url = f'{base_url}{self.name.upper()}{year}'
            if year >= 2018:
                yield scrapy.Request(
                    url, callback=self.parse_day, cb_kwargs={'year': year})
            else:
                yield scrapy.Request(
                    url,
                    callback=self.parse_paper_list,
                    cb_kwargs={'year': year})

    def parse_day(self, response, year):
        # Now we navigate to the Day page.
        # Get all the days listed there using the xpath.
        # extract() generates a list of all matched elements.
        day_url_list = response.xpath(
            "//div[@id='content']/dl/dd/a/@href").extract()

        # Traverse every day
        for day_url in day_url_list:

            # Exclude the Day-aLL hyperlink to avoid redundancy.
            if "day=all" in day_url:
                continue

            url = response.urljoin(day_url)
            yield scrapy.Request(
                url,
                callback=self.parse_paper_list,
                cb_kwargs=response.cb_kwargs)

    def parse_paper_list(self, response, year):
        # Now we have all the papers.
        paper_url_list = response.xpath(
            "//div[@id='content']/dl/dt[@class='ptitle']/a/@href").extract()

        for paper_url in paper_url_list:
            url = response.urljoin(paper_url)

            # for each paper, navigate to its detail page
            yield scrapy.Request(
                url,
                callback=self.parse_paper_details,
                cb_kwargs=response.cb_kwargs)

    def parse_paper_details(self, response, year):
        title = inspect.cleandoc(
            response.xpath("//div[@id='papertitle']/text()").get())
        pdf_url = response.urljoin(
            response.xpath("//div[@id='content']/dl/dd/a[1]/@href").get())
        authors = inspect.cleandoc(
            response.xpath("//div[@id='authors']/b/i/text()").get())
        abstract = inspect.cleandoc(
            response.xpath("//div[@id='abstract']/text()").get())

        return Paper(
            conference=self.name.upper(),
            year=year,
            title=title,
            authors=authors,
            abstract=abstract,
            pdf_url=pdf_url)


class CVPRSpider(CVFSpider):
    name = 'cvpr'


class ICCVSpider(CVFSpider):
    name = 'iccv'


class WACVSpider(CVFSpider):
    name = 'wacv'

    def start_requests(self):
        base_url = 'https://openaccess.thecvf.com/'
        for year in self.years:
            url = f'{base_url}{self.name.upper()}{year}'
            yield scrapy.Request(
                url, callback=self.parse_paper_list, cb_kwargs={'year': year})


class ECCVSpider(BaseSpider):

    name = 'eccv'
    start_urls = [
        "https://www.ecva.net/papers.php",
    ]

    def parse(self, response):

        for year in self.years:
            if year < 2018 or year % 2 != 0:
                self.logger.warn(f'ECCV {year} is not supported')
                continue
            buttons = response.xpath("//button[@class='accordion']")
            for item in buttons:
                if str(year) in item.get():
                    paper_url_list = item.xpath(
                        "following-sibling::div[1]/div/dl/dt/a/@href").extract(
                        )
                    break

            for paper_url in paper_url_list:
                url = response.urljoin(paper_url)
                yield scrapy.Request(
                    url,
                    callback=self.parse_paper_details,
                    cb_kwargs={'year': year})

    def parse_paper_details(self, response, year):
        title = inspect.cleandoc(
            response.xpath("//div[@id='papertitle']/text()").get())
        pdf_url = response.urljoin(
            response.xpath("//div[@id='content']/dl/dd/a[1]/@href").get())
        authors = inspect.cleandoc(
            response.xpath("//div[@id='authors']/b/i/text()").get())
        abstract = inspect.cleandoc(
            response.xpath("//div[@id='abstract']/text()").get()).replace(
                '\n', ' ').rstrip()

        return Paper(
            conference=self.name.upper(),
            year=year,
            title=title,
            authors=authors,
            abstract=abstract,
            pdf_url=pdf_url)


class NeurIPSSpider(BaseSpider):
    name = 'neurips'

    def start_requests(self):
        for year in self.years:
            url = f'https://papers.neurips.cc/paper/{year}'
            yield scrapy.Request(
                url, callback=self.parse_paper_list, cb_kwargs={'year': year})

    def parse_paper_list(self, response, year):
        paper_url_list = response.xpath(
            "//div[@class='container-fluid']/div[@class='col']/ul/li/a/@href"
        ).extract()

        for paper_url in paper_url_list:
            url = response.urljoin(paper_url)
            yield scrapy.Request(
                url,
                callback=self.parse_paper_details,
                cb_kwargs=response.cb_kwargs)

    def parse_paper_details(self, response, year):
        title = inspect.cleandoc(
            response.xpath("//div[@class='col']/h4/text()").get())
        authors = inspect.cleandoc(
            response.xpath(
                "//div[@class='col']/p[position()=2]/i/text()").get())
        abstract = response.xpath(
            "//div[@class='col']/p[position()=4]/text()").get()
        if not abstract:
            abstract = response.xpath(
                "//div[@class='col']/p[position()=3]/text() | //div[@class='col']/p[position()=3]/span/text() | //div[@class='col']/pre/code/text()"
            ).get()
        abstract = inspect.cleandoc(abstract).replace('\n', ' ')

        pdf_url = response.urljoin(
            response.xpath(
                "//div[@class='col']/div/a[text()='Paper']/@href").get())

        return Paper(
            conference='NeurIPS',
            year=year,
            title=title,
            authors=authors,
            abstract=abstract,
            pdf_url=pdf_url)


class AAAISpider(BaseSpider):

    name = 'aaai'

    def start_requests(self):
        for year in self.years:
            url = f'https://aaai.org/Library/AAAI/aaai{str(year)[2:]}contents.php'
            if year >= 2020:
                yield scrapy.Request(
                    url,
                    callback=self.parse_track_list,
                    cb_kwargs={'year': year})
            else:
                yield scrapy.Request(
                    url,
                    callback=self.parse_paper_list_before2020,
                    cb_kwargs={'year': year})

    def parse_track_list(self, response, year):
        elements = response.xpath("//div[@class='content']/ul/li/a")
        for elem in elements:
            subtitle = elem.xpath('text()').get()
            # exclude special programs and student tracks
            if 'Technical Track' in subtitle:
                url = response.urljoin(elem.xpath('@href').get())
                yield scrapy.Request(
                    url,
                    callback=self.parse_paper_list,
                    cb_kwargs=response.cb_kwargs)

    def parse_paper_list(self, response, year):
        paper_url_list = response.xpath(
            "//div[@id='content']/div[@id='right']/div[@id='box6']/div[@class='content']/p/a[1]/@href"
        ).extract()

        for paper_url in paper_url_list:
            url = response.urljoin(paper_url)
            yield scrapy.Request(
                url,
                callback=self.parse_paper_details,
                cb_kwargs=response.cb_kwargs)

    def parse_paper_list_before2020(self, response, year):
        lines = response.xpath(
            '//div[@id="content"]/div/div/div/h4/text() | //div[@id="content"]/div/div/div/p[@class="left"]/a[1]/@href'
        ).extract()
        exclude_list = [
            'Innovative Applications of Artificial Intelligence',
            'Educational Advances in Artificial Intelligence Symposium',
            'Senior Member', 'Student Abstracts', 'Doctoral Consortium',
            'What\'s Hot', 'Demonstration', 'IAAI', 'EAAI'
        ]

        def contains(string, query_list):
            for q in query_list:
                if q in string:
                    return True
            return False

        paper_urls = []
        for line in lines:
            if line.startswith('http'):
                paper_urls.append(line)
            if contains(line, exclude_list):
                break

        for url in paper_urls:
            yield scrapy.Request(
                url,
                callback=self.parse_paper_details,
                cb_kwargs=response.cb_kwargs)

    def parse_paper_details(self, response, year):
        title = inspect.cleandoc(response.xpath("//article/h1/text()").get())
        authors = inspect.cleandoc(",".join(
            response.xpath(
                "//ul[@class='authors']/li/span[@class='name']/text()").
            extract()).replace("\t", "").replace("\n", ""))
        abstract = inspect.cleandoc("".join(
            response.xpath(
                "//section[@class='item abstract']/p/text() | //section[@class='item abstract']/text()"
            ).extract()).replace("\t", "").replace("\n", ""))

        pdf_url = response.xpath(
            "//div[@class='entry_details']/div[@class='item galleys']/ul/li/a/@href"
        ).get()

        return Paper(
            conference=self.name.upper(),
            year=year,
            title=title,
            authors=authors,
            abstract=abstract,
            pdf_url=pdf_url)


class IJCAI(BaseSpider):

    name = 'ijcai'

    def start_requests(self):
        for year in self.years:
            if year < 2017:
                self.logger.warn('IJCAI {year} is not supported')
                continue
            url = f'https://www.ijcai.org/proceedings/{year}'
            yield scrapy.Request(
                url, callback=self.parse_paper_list, cb_kwargs={'year': year})

    def parse_paper_list(self, response, year):
        paper_url_list = response.xpath(
            "//div[@class='paper_wrapper']/div[@class='details']/a[2]/@href"
        ).extract()

        for paper_url in paper_url_list:
            url = response.urljoin(paper_url)
            yield scrapy.Request(
                url,
                callback=self.parse_paper_details,
                cb_kwargs=response.cb_kwargs)

    def parse_paper_details(self, response, year):
        title = inspect.cleandoc(
            response.xpath("//div[@class='row'][1]/div/h1/text()").get())
        authors = inspect.cleandoc(
            response.xpath("//div[@class='row'][1]/div/h2/text()").get())
        abstract = inspect.cleandoc(
            response.xpath("//div[@class='row'][3]/div/text()").get()).replace(
                '\n', ' ')
        pdf_url = response.xpath("//div[@class='btn-container']/a/@href").get()

        return Paper(
            conference=self.name.upper(),
            year=year,
            title=title,
            authors=authors,
            abstract=abstract,
            pdf_url=pdf_url)


class ICLRSpider(BaseSpider):

    name = 'iclr'

    def start_requests(self):
        for year in self.years:
            if year >= 2022:
                for paper_type in ['Oral', 'Spotlight', 'Poster']:
                    url = f'https://api.openreview.net/notes?content.venue=ICLR+2022+{paper_type}&details=original,directReplies&offset=0&limit=1000&invitation=ICLR.cc/{year}/Conference/-/Blind_Submission'
                    yield scrapy.Request(
                        url,
                        callback=self.parse_paper_details,
                        cb_kwargs={
                            'year': year,
                            'limit': 1000,
                            'offset': 0
                        })
            else:
                url = f'https://api.openreview.net/notes?invitation=ICLR.cc/{year}/Conference/-/Blind_Submission&details=directReplies&limit=1000&offset=0'
                yield scrapy.Request(
                    url,
                    callback=self.parse_paper_details,
                    cb_kwargs={
                        'year': year,
                        'limit': 1000,
                        'offset': 0
                    })

    def parse_paper_details(self, response, year, limit, offset):
        result = response.json()
        if result['count'] > limit + offset:
            offset += limit
            if year >= 2022:
                url = f'https://api.openreview.net/notes?content.venue=ICLR+{year}+Submitted&details=replyCount&offset={offset}&limit={limit}&invitation=ICLR.cc/{year}/Conference/-/Blind_Submission'
            else:
                url = f'https://api.openreview.net/notes?invitation=ICLR.cc/{year}/Conference/-/Blind_Submission&details=directReplies&limit={limit}&offset={offset}'
            yield scrapy.Request(
                url,
                callback=self.parse_paper_details,
                cb_kwargs={
                    'year': year,
                    'limit': limit,
                    'offset': offset
                })

        for paper_info in result['notes']:
            accepted = True
            if year < 2022:
                for reply in paper_info['details']['directReplies'][::-1]:
                    if 'decision' in reply['content']:
                        if reply['content']['decision'] == 'Reject':
                            accepted = False
                            break
            if not accepted:
                continue
            content = paper_info['content']
            title = content['title']
            authors = ', '.join(content['authors'])
            abstract = content['abstract'].replace('\n', ' ').replace('\0', '')
            pdf_url = 'https://openreview.net' + content['pdf']
            code_url = None
            if 'code' in content and content['code']:
                matched = re.search(r'\(https://github.com/(.*?)\)',
                                    content['code'])
                if matched:
                    code_url = matched.group(0)[1:-1]
            yield Paper(
                conference=self.name.upper(),
                year=year,
                title=title,
                authors=authors,
                abstract=abstract,
                pdf_url=pdf_url,
                code_url=code_url)


# class ICLRSpider(BaseSpider):

#     name = 'iclr'

#     def start_requests(self):

#         try:
#             from scrapy_selenium import SeleniumRequest
#             from selenium.webdriver.common.by import By
#             from selenium.webdriver.support import expected_conditions as EC
#         except ImportError:
#             raise ImportError('ICLR spider requires selenium to be installed')

#         for year in self.years:
#             if year == 2022:
#                 presentation_types = [
#                     'oral-submissions', 'spotlight-submissions',
#                     'poster-submissions'
#                 ]
#             elif year >= 2020:
#                 presentation_types = [
#                     'oral-presentations', 'spotlight-presentations',
#                     'poster-presentations'
#                 ]
#             elif year == 2019:
#                 presentation_types = [
#                     'oral-presentations', 'poster-presentations'
#                 ]
#             elif year == 2018:
#                 presentation_types = [
#                     'accepted-oral-papers', 'accepted-poster-papers'
#                 ]
#             else:
#                 self.logger.warn('ICLR {year} is not supported')
#                 continue

#             url = f'https://openreview.net/group?id=ICLR.cc/{year}/Conference'

#             for pre_type in presentation_types:
#                 yield SeleniumRequest(
#                     url=url,
#                     callback=self.parse_paper_list,
#                     cb_kwargs={
#                         'year': year,
#                         'pre_type': pre_type
#                     },
#                     wait_time=15,
#                     wait_until=EC.visibility_of_element_located(
#                         (By.XPATH,
#                          '//ul[@class="list-unstyled submissions-list"]/li')),
#                     script=
#                     f"document.querySelector('a[aria-controls=\"{pre_type}\"]').click()",
#                     dont_filter=True)

#     def parse_paper_list(self, response, year, pre_type):
#         urls = response.xpath(
#             f'//div[@id="{pre_type}"]/ul[@class="list-unstyled submissions-list"]/li/h4/a[1]/@href'
#         ).extract()
#         for url_part in urls:
#             yield scrapy.Request(
#                 url='https://openreview.net/' + url_part,
#                 callback=self.parse_paper_details,
#                 cb_kwargs={'year': year})

#     def parse_paper_details(self, response, year):

#         title = response.xpath(
#             '//div[@class="forum-container"]/div/div[1]/h2/text()').get()
#         pdf_url = 'https://openreview.net/' + response.xpath(
#             '//div[@class="note"]/div/h2/a/@href').get()
#         authors = ', '.join(
#             response.xpath('//div[@class="meta_row"]/h3/a/text()').extract())
#         abstract = inspect.cleandoc(
#             response.xpath(
#                 "//div[@class='note-content']/div/strong[text()='Abstract']/following-sibling::span/text()"
#             ).get()).replace('\n', ' ')
#         code_region = response.xpath(
#             "//div[@class='note-content']/div/strong[text()='Code']/following-sibling::span/text()"
#         ).get()
#         code_url = None
#         if code_region:
#             matched = re.search(r'\(https://github.com/(.*?)\)', code_region)
#             if matched:
#                 code_url = matched.group(0)[1:-1]
#         return Paper(
#             conference=self.name.upper(),
#             year=year,
#             title=title,
#             authors=authors,
#             abstract=abstract,
#             pdf_url=pdf_url,
#             code_url=code_url)


class PMLRSpider(BaseSpider):

    start_urls = ['https://proceedings.mlr.press/']

    def parse(self, response):
        for year in self.years:
            vol = response.xpath(
                f'//ul[@class="proceedings-list"]/li[text()=" Proceedings of {self.name.upper()} {year}"]/a/@href | //ul[@class="proceedings-list"]/li[text()=" {self.name.upper()} {year} Proceedings"]/a/@href'
            ).get()
            if not vol:
                self.logger.warn(
                    f'{self.name.upper()} {year} is not supported')
            url = response.urljoin(vol)
            yield scrapy.Request(
                url, callback=self.parse_paper_list, cb_kwargs={'year': year})

    def parse_paper_list(self, response, year):
        abs_urls = response.xpath(
            '//div[@class="paper"]/p[@class="links"]/a[text()="abs"]/@href'
        ).extract()
        for url in abs_urls:
            yield scrapy.Request(
                url,
                callback=self.parse_paper_details,
                cb_kwargs=response.cb_kwargs)

    def parse_paper_details(self, response, year):
        title = response.xpath('//article/h1/text()').get()
        authors = response.xpath(
            '//article/span[@class="authors"]/text()').get().replace(
                '\xa0', ' ')
        abstract = inspect.cleandoc(
            response.xpath(
                '//article/div[@id="abstract"]/text()').get()).replace(
                    '\n', ' ')
        pdf_url = response.xpath(
            '//div[@id="extras"]/ul/li/a[text()="Download PDF"]/@href').get()
        return Paper(
            conference=self.name.upper(),
            year=year,
            title=title,
            authors=authors,
            abstract=abstract,
            pdf_url=pdf_url)


class ICMLSpider(PMLRSpider):

    name = 'icml'


# class MmScrapySpider(BaseSpider):
#     name = 'mm'
#     # start_urls = [
#     #     "https://dl.acm.org/pb/widgets/proceedings/getProceedings?widgetId=517fcc12-7ff3-4236-84f8-899a672b4a79&pbContext=;taxonomy:taxonomy:conference-collections;topic:topic:conference-collections>mm;page:string:Proceedings;wgroup:string:ACM Publication Websites;csubtype:string:Conference;ctype:string:Conference Content;website:website:dl-site;pageGroup:string:Publication Pages&ConceptID=119833",
#     # ]
#     start_urls = [
#         "https://dl.acm.org/pb/widgets/proceedings/getProceedings?widgetId=517fcc12-7ff3-4236-84f8-899a672b4a79&pbContext=;taxonomy:taxonomy:conference-collections;topic:topic:conference-collections>mm;page:string:Proceedings;wgroup:string:ACM Publication Websites;csubtype:string:Conference;ctype:string:Conference Content;website:website:dl-site;pageGroup:string:Publication Pages&ConceptID=119833",
#     ]
#     base_url = "https://dl.acm.org"
#     download_delay = 3

#     def parse(self, response):
#         received_data = json.loads(response.text)
#         for conf in self.wanted_conf:
#             for conf_data in received_data['data']['proceedings']:
#                 if conf_data['title'].split(":")[0][-2:] == conf[-2:]:
#                     link = conf_data['link']
#                     break
#             url = self.base_url + link
#             yield scrapy.Request(url, callback=self.parse_session_list)

#     def parse_session_list(self, response):
#         session_list = response.xpath(
#             "//div[@class='accordion sections']/div[@class='accordion-tabbed rlist']/div/a/@href"
#         ).extract()

#         for session in session_list:
#             doi = re.search(pattern=r'10(.+?)\?', string=session)[0][:-1]
#             tocHeading = session.split("=")[1]
#             url = "https://dl.acm.org/pb/widgets/lazyLoadTOC?tocHeading={}&widgetId=f51662a0-fd51-4938-ac5d-969f0bca0843&doi={}&pbContext=;" \
#                   "article:article:doi\:{};" \
#                   "taxonomy:taxonomy:conference-collections;" \
#                   "topic:topic:conference-collections>mm;" \
#                   "wgroup:string:ACM Publication Websites;" \
#                   "groupTopic:topic:acm-pubtype>proceeding;" \
#                   "csubtype:string:Conference Proceedings;" \
#                   "page:string:Book Page;" \
#                   "website:website:dl-site;" \
#                   "ctype:string:Book Content;journal:journal:acmconferences;" \
#                   "pageGroup:string:Publication Pages;" \
#                   "issue:issue:doi\:{}".format(tocHeading, doi, doi, doi)

#             yield scrapy.Request(url, callback=self.parse_paper_list)

#     def parse_paper_list(self, response):
#         doi_list = response.xpath(
#             "//div[@class='issue-item clearfix']/div/div/h5/a/@href").extract(
#             )

#         for doi in doi_list:
#             url = self.base_url + doi
#             yield scrapy.Request(url, callback=self.parse_paper)

#     @staticmethod
#     def extract_data(response):

#         title = response.xpath(
#             "//div[@class='article-citations']/div[@class='citation']/div[@class='border-bottom clearfix']/h1/text()"
#         ).get()
#         clean_title = re.sub(r'\W+', ' ', title).lower()

#         authors = inspect.cleandoc(",".join(
#             response.xpath(
#                 "//div[@class='article-citations']/div[@class='citation']/div[@class='border-bottom clearfix']/div[@id='sb-1']/ul/li[@class='loa__item']/a/@title"
#             ).extract()))

#         abstract = inspect.cleandoc(
#             response.xpath(
#                 "//div[@class='abstractSection abstractInFull']/p/text()").get(
#                 ))
#         conf = response.xpath(
#             "//div[@class='article-citations']/div[@class='citation']/div[@class='border-bottom clearfix']/div[@class='issue-item__detail']/a/@title"
#         ).get().split(":")[0]
#         pdf_url = ""

#         return conf, title, pdf_url, clean_title, authors, abstract