from collections import OrderedDict 
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
import ftfy
import unicodedata
import lxml.etree as etree


class OrderedDefaultListDict(OrderedDict):
    def __missing__(self, key):
        self[key] = value = [] 
        return value

none_to_empty_str = lambda s : "" if s is None else s 

def fmt(s):
    return unicodedata.normalize('NFKD', ftfy.fix_text(s))

class TextArticle:

    def __init__(self, text, name, article_id):
        self.article_id = str(article_id)
        self.text = text
        self.name = name

    def get_pmcid(self):
        return self.article_id

    def to_raw_str(self, fields=None, join_para_on=" <p> ", join_sections_on="\n\n"):
        return self.text

# TODO: not that this will ever happen, but article reading, writibng, and conversion should all be separaterd
class Article:
    '''
    Container class for articles. Responsible for consuming and parsing
    XMLs, and providing access to these.
    '''

    # front matter/meta data and body indices.
    FRONT_IDX, BODY_IDX = 0, 1

    def __init__(self, xml_path, use_plain_text = False):
        self.id = xml_path
        self.use_plain_text = use_plain_text
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

    def get_pmcid(self):
        return re.match('.*PMC([0-9]+).nxml', self.id).group(1)
    
    def format_xmlns(self, s, el):
        nsmap_inverse = { v: k for k, v in el.nsmap.items() }
        # universal non-specified namespace as per:
        #     https://www.w3.org/TR/xml-names/
        nsmap_inverse['http://www.w3.org/XML/1998/namespace'] = 'xml'
        prefix, suffix = s[1:].split('}', 1)
        ns = nsmap_inverse.get(prefix, '')
        if not ns:
          pass
        return '{}:{}'.format(ns, suffix)

    def gen_plain_text(self, fname = None):
        """
        Get the plain text version of the NXML file.
        
        @param fname     is the full path/location of the XML file.
        @return txt_out  is a string that is the plain text version of the xml.
        @return all_sections is a dictionary with section names and their character offsets.
        """
        fname = fname or self.id
        txt_out = ''   
        newline_tags = ['p', 'sec', 'title', 'td']
        collected_tags = ['abstract', 'body']
        collect_txt = False
        tag_list = []
        all_sections = {}
    
        for e, el in etree.iterparse(open(fname, 'rb'), ('start', 'end')):
            if e == 'start':
                tag = el.tag
                if el.tag.startswith('{'):
                  tag = self.format_xmlns(tag, el)
                
                if tag in collected_tags or ((tag == 'sec' or tag == 'title') and el.text != None):
                    tag_list.append(tag if tag in collected_tags else el.text)
                    all_sections[".".join(tag_list)] = {'start': len(txt_out)}
                       
                if tag in collected_tags:
                    collect_txt = True
                    txt_out += '<{}>\n'.format(tag.upper())
            
                if el.text:
                    if collect_txt:  
                      txt_out += el.text
            
            elif e == 'end':
                tag = el.tag
                if el.tag.startswith('{'):
                    tag = self.format_xmlns(tag, el)
            
                if collect_txt and tag in newline_tags:
                    txt_out += '\n'
                    
                if (tag in collected_tags or tag == 'sec') and len(tag_list) > 0:
                    all_sections[".".join(tag_list)]['end'] = len(txt_out)
                    del tag_list[-1]
            
                # wrap tail in a special html tag since the start index is no longer linked to the opening tag
                if el.tail:
                    if collect_txt:
                      txt_out += el.tail
            
                if tag in collected_tags:
                    collect_txt = False
                    
        return txt_out, all_sections

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
        if self.use_plain_text:
            return self.gen_plain_text(self.id)[0]
        
        if fields is None:
            fields = self.article_dict.keys()

        out_str = []
        title = 'TITLE: ' + self.get_title() + "\n\n  "
        for field in fields: 
            texts = [none_to_empty_str(s) for s in self.article_dict[field]]
            field_text = join_para_on.join(texts)
            out_str.append(field.upper() + ": " + field_text)
            
        return title + join_sections_on.join(out_str)

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

