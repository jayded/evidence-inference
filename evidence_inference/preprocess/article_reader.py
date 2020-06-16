

from collections import OrderedDict 
class OrderedDefaultListDict(OrderedDict):
    def __missing__(self, key):
        self[key] = value = [] 
        return value

import xml.etree.ElementTree as ET
from html.parser import HTMLParser
import ftfy
import unicodedata
import lxml.etree as etree



none_to_empty_str = lambda s : "" if s is None else s 

def fmt(s):
    return unicodedata.normalize('NFKD', ftfy.fix_text(s))

class Article:
    '''
    Container class for articles. Responsible for consuming and parsing
    XMLs, and providing access to these.
    '''

    # front matter/meta data and body indices.
    FRONT_IDX, BODY_IDX = 0, 1

    def __init__(self, xml_path):
        self.id = xml_path
        self.article_tree = ET.parse(xml_path)
        self.article_root = self.article_tree.getroot()
        self.article_dict = OrderedDefaultListDict()

        self.article_meta = self.article_root[self.FRONT_IDX].findall("article-meta")[0]
        self.parse_article_abstract()
        
        try:
            self.body = self.article_root[self.BODY_IDX]
            self.parse_article_body()
        except:
            # this means that the article here has only an abstract.
            self.body = []

    def __str__(self):
        return self.get_title()
    
    def get_title(self):
        # note that we return an empty string if the title is missing.
        return self.article_meta.findall("title-group")[0].findall("article-title")[0].text or ""

    def get_abstract(self, structured=True):
        abstract_keys = self._get_abstract_keys()

        if structured:
            return OrderedDict(zip(abstract_keys, 
                                  [self.article_dict[k] for k in abstract_keys]))
        else:
            return self.to_raw_str(fields=abstract_keys)
       
    def ti_ab_str(self):
        return "TITLE: " + self.get_title() \
                + "\n\n" + "ABSTRACT \n" + self.get_abstract(structured=False)

    def to_raw_str(self, fields=None, join_para_on=" <p> ", join_sections_on="\n\n"):
        '''
        Generate an unstructured string representation of the article, or of the
        subset of the article specified by the optional 'fields' argument.
        '''
        if fields is None:
            fields = self.article_dict.keys()

        out_str = []
        for field in fields: 
            texts = [none_to_empty_str(s) for s in self.article_dict[field]]
            field_text = join_para_on.join(texts)
            out_str.append(field.upper() + ":\n" + field_text)
            
        return join_sections_on.join(out_str)

    def _get_abstract_keys(self):
        return [k for k in self.article_dict.keys() if "abstract" in k]

    def _get_section_name(self, section_element):
        title_elements = section_element.findall("title")
        if len(title_elements) == 0:
            return None
        return title_elements[0].text 

    def parse_article_abstract(self):
        try:
            abstract_element = self.article_meta.findall("abstract")[0]
            self.parse_element(abstract_element, "abstract")
        except:
            self.article_dict["abstract"].append("") # no abstract!

    def parse_article_body(self):
        start_node = self.body
        self.parse_element(start_node, "body")

    def parse_element(self, start_node, parent_section_str=None):
        ''' 
        Return a dictionary mapping (body) section names to 
        texts. The latter are stored as lists of paragraphs,
        as per the <p> tags.


        start_node is an Element, which is a list of Elements. 
        These may include 'sec' Elements, 'p' and 'title' Elements,
        or some mix. This will recursively parse such elements,
        and populate the 'article_dict' class var.
        '''
        parser = MyHTMLParser()      
        parser_table = TableHTMLParser()
    
        section_name = self._get_section_name(start_node)
        if parent_section_str is not None:
            if section_name is None:
                section_name = parent_section_str
            else:
                section_name = "{}.{}".format(parent_section_str, none_to_empty_str(section_name))
        
        for element in start_node:
            section_type = element.tag 
            
            if section_type == "sec":
                self.parse_element(element, parent_section_str=section_name) 
            elif section_type == "p":
                txt = ET.tostring(element).decode("utf-8")
                parser.feed(txt)
                txt = fmt(parser.get_data())
                self.article_dict[section_name].append(txt)
            elif section_type == "table-wrap":
                txt = ET.tostring(element).decode("utf-8")
                parser_table.feed(txt)
                to_app = parser_table.get_data()
                self.article_dict[section_name].append(to_app)
            elif section_type == 'title':
                # we don't need to include title headers because we already extract, 
                # and use them to join strings (including would only double them).
                continue
            else:
                txt = ET.tostring(element).decode("utf-8")
                parser.feed(txt)
                txt = fmt(parser.get_data())
                self.article_dict[section_name].append(txt)

# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):
    data = ""
        
    def handle_starttag(self, tag, attrs):
        return None
        #self.data += " " * (len(tag) + self.get_all_lengths(attrs) + 2) # start and end 

    def handle_endtag(self, tag):
        return None

    def handle_data(self, data):
        self.data += data
    
    def get_data(self):
        tmp = self.data
        self.data = ""
        return tmp
    
# create a subclass and override the handler methods
class TableHTMLParser(HTMLParser):
    data = " "
        
    def handle_starttag(self, tag, attrs):
        return None
        #self.data += " " * (len(tag) + self.get_all_lengths(attrs) + 2) # start and end 

    def handle_endtag(self, tag):
        return None

    def handle_data(self, data):
        self.data += data + " "
    
    def get_data(self):
        tmp = self.data
        self.data = ""
        return tmp

