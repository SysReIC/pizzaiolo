import sys
import owlready2
import math
from random import *
import numpy as np
import os 
import shutil
import pandas as pd
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
from sklearn.model_selection import train_test_split

from collections import OrderedDict

class Pizzaiolo():

    # --------------------------------------------- STATIC ------------------------------------------------

    CSV_DIR = "csv"
    CSV_FILENAME = "pizzaiolo_dataset.csv"
    CSV_TRAIN_FILENAME = "pizzaiolo_train.csv"
    CSV_VALID_FILENAME = "pizzaiolo_valid.csv"
    CSV_TEST_FILENAME = "pizzaiolo_test.csv"
    REF_COLUMN_NAME = "ref"
    TYPE_COLUMN_NAME = "type"
    HASBASE_COLUMN_NAME = "hasBase"
    HASCOUNTRY_COLUMN_NAME = "hasCountryOfOrigin"
    MULTILABEL_COLUMN_NAME = "multilabel"

    ONTOLOGY_DIR = "ontology"
    ONTOLOGY_NAME = "pizzaiolo"

    IMAGES_DIR = "images"
    IMAGES_SUFFIX = ".png"
    
    FLAG_SIZE = (30, 30)

    LABELS_DIR = "labels"
    BOXES_SUFFIX = "_bboxes.json"
    CONTOURS_SUFFIX = "_contours.json"
    SEGMENTATION_SUFFIX = "_segmentation.txt"

    ENTITIES_FILENAME = "concepts.json"

    ENTITY_COLORS = {
        'None': (255, 255, 255),
        
        'DeepPanBase': (136, 74, 63),
        'ThinAndCrispyBase': (249, 157, 42),
        
        'ParmesanTopping': (232, 211, 141),
        'SlicedTomatoTopping': (255, 99, 69),
        'OnionTopping': (243, 122, 236),
        'MushroomTopping': (216, 174, 134),
        'PeperoniSausageTopping': (233, 88, 89),
        'HamTopping': (180, 42, 42),
        'GreenPepperTopping': (172, 223, 75),
        'PeperonataTopping': (0, 255, 0),
        'RocketTopping': (104, 165, 44),
        'SpinachTopping': (91, 200, 30),
        'JalapenoPepperTopping': (226, 31, 39),
        'AnchovyTopping': (97, 118, 200),
        'PrawnTopping': (255, 100, 83),
        'OliveTopping': (55, 55, 55),
        'SultanaTopping': (104, 183, 56),
        'GarlicTopping': (216, 200, 61),
        
        'America': (0, 255, 0),
        'England': (0, 255, 0),
        'France': (0, 255, 0),
        'Italy': (0, 255, 0),
    }

    @staticmethod
    def display(img_list:list, titles:list[str]=None, figsize:(int,int)=None, fontsize:(int,int)=None):
        """
        Displays a list of images
        
        Parameters:
            - img_list : list of images
            - titles : list[string] of titles
            - figsize : (width, height)
        """
        
        plt.figure(figsize=figsize)
        for i in range(len(img_list)):
            plt.subplot(1, len(img_list), i+1)
            if not titles is None:
                plt.title(titles[i], fontsize=fontsize)
            plt.imshow(img_list[i])
            plt.axis('off')
        plt.show()

    @staticmethod
    def drawAllBoxes(img, toppings_boxes, thickness=2):
        """
        Draws the bounding boxes on the img
        
        Parameters:
        - img: the image to draw on
        - toppings_boxes: dictionary of bounding boxes
        - thickness: the thickness of the bounding boxes
        """
        for t, boxes in toppings_boxes.items():
            img = Pizzaiolo.drawBoxes(img, boxes, color=Pizzaiolo.getColorFor(t), thickness=thickness)
        return img

    @staticmethod
    def drawAllContours(img, toppings_contours, thickness=cv2.FILLED):
        for t, contours in toppings_contours.items():
            img = Pizzaiolo.drawContours(img, contours, color=Pizzaiolo.getColorFor(t), thickness=thickness)
        return img

    @staticmethod
    def drawBoxes(pil_img, boxes, color=(0,255,0), thickness=1):
        with_boxes = np.array(pil_img)
        for x,y,w,h in boxes:
            cv2.rectangle(with_boxes, (x,y), (x+w, y+h), color, thickness)
        return Image.fromarray(with_boxes).convert("RGB")

    @staticmethod
    def drawContours(pil_img, contours, color=(0,255,0), thickness=cv2.FILLED):
        im_with_contours = cv2.drawContours(
            image= np.array(pil_img),
            contours=contours,
            contourIdx=-1,
            color=color,
            thickness=thickness)
        return Image.fromarray(im_with_contours).convert("RGB")

    @staticmethod
    def _findMaskContours(mask, min_perimeter=20):
        new_mask = np.zeros_like(mask, dtype='uint8')
        new_mask[mask>0] = 255
        new_mask = Image.fromarray(new_mask).convert("RGB")
        cv_mask = np.array(new_mask)
        gray = cv2.cvtColor(cv_mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_contours = []
        filtered_hierarchy = []
        for i, c in enumerate(contours):
            if cv2.arcLength(c, True) >= min_perimeter:
                filtered_contours.append(c)
                filtered_hierarchy.append(hierarchy[0][i])
        return filtered_contours, filtered_hierarchy

    @staticmethod
    def _findApproxContours(contours, min_perimeter=20, threshold=0.01):
        approx = []
        for c in contours:
            peri = cv2.arcLength(c, True)
            if peri >= min_perimeter:
                approx.append(cv2.approxPolyDP(c, peri * threshold, True))
        return approx

    @staticmethod
    def _findBoxes(contours, hierarchy):
        filtered_contours = []
        filtered_contours = []
        for i, h in enumerate(hierarchy):
            next, previous, first_child, parent = h
            if parent == -1: filtered_contours.append(contours[i])
        boxes = []
        for c in filtered_contours:
            boxes.append(cv2.boundingRect(c))
        return boxes

    @staticmethod
    def getColorFor(topping_str):
        return Pizzaiolo.ENTITY_COLORS[topping_str.split('.')[-1]]

    @staticmethod
    def _getMaskedImage(img, mask):
        """
        Returns:
            - a PIL Image which is img masked with mask
        """
        new_array = np.array(img)
        new_array[mask==0, :] = 0
        return Image.fromarray(new_array)

    @staticmethod
    def loadBoxes(dataset_dir, json_filename):
        file_name = os.path.join(dataset_dir, Pizzaiolo.LABELS_DIR, json_filename)
        toppings_boxes = json.load(open(file_name))
        for t, boxes in toppings_boxes.items():
            for i, b in enumerate(boxes):
                boxes[i] = np.array(b)
        return toppings_boxes

    @staticmethod
    def loadContours(dataset_dir, json_filename):
        file_name = os.path.join(dataset_dir, Pizzaiolo.LABELS_DIR, json_filename)
        toppings_contours = json.load(open(file_name))
        for t, contours in toppings_contours.items():
            for i, c in enumerate(contours):
                contours[i] = np.array(c)
        return toppings_contours

    @staticmethod
    def loadImage(dataset_dir, image_filename):
        file_name = os.path.join(dataset_dir, Pizzaiolo.IMAGES_DIR, image_filename)
        return Image.open(file_name)

    @staticmethod
    def loadCSV(dataset_dir, csv_filename):
        df = pd.read_csv(os.path.join(dataset_dir, Pizzaiolo.CSV_DIR, csv_filename), index_col=Pizzaiolo.REF_COLUMN_NAME)
        df['hasTopping'] = df['hasTopping'].apply(eval)
        try:
            df[Pizzaiolo.MULTILABEL_COLUMN_NAME] = df[Pizzaiolo.MULTILABEL_COLUMN_NAME].apply(eval)
        except: pass
            
        return df
    
    @staticmethod
    def loadOntology(dataset_dir):
        ontology_dir = os.path.join(dataset_dir, Pizzaiolo.ONTOLOGY_DIR)
        onto_filename = os.listdir(ontology_dir)[0]
        onto_filename = os.path.join(ontology_dir, onto_filename)
        ontology = owlready2.get_ontology(onto_filename).load()
        ontology.name = Pizzaiolo.ONTOLOGY_NAME
        return ontology
    
    @staticmethod
    def loadSegmentation(dataset_dir, segmentation_filename):
        """
        Returns:
        - a numpy array
        """
        file_name = os.path.join(dataset_dir, Pizzaiolo.LABELS_DIR, segmentation_filename)
        return np.loadtxt(file_name, dtype='uint')

    @staticmethod
    def loadToppingTypes(dataset_dir):
        toppings_path = os.path.join(dataset_dir, Pizzaiolo.LABELS_DIR, Pizzaiolo.ENTITIES_FILENAME)
        f = open(toppings_path)
        toppings = json.load(f)
        f.close()
        ids_toppings = {}
        for id, t in toppings.items():
            ids_toppings[eval(id)] = t
        return OrderedDict(sorted(ids_toppings.items()))
    
    @staticmethod
    def splitDataset(delivery_dir, train_size=0.8, valid_size=None, test_size=None):
        """
        Splits a pizzaiolo dataset : generates a train and valid (and optional test) dataframes
        
        Parameters:
        - delivery_dir: the location of the dataset
        """
        
        if valid_size is None:
            valid_size = 1 - train_size
        if test_size is None:
            total = train_size + valid_size
        else:
            inter_size = valid_size + test_size
            total = train_size + inter_size
        if total != 1:
            raise Exception("train (%g)+ valid (%g) + test (%g) should be equal to 1" % (train_size, valid_size, test_size))

        csv_dir = os.path.join(delivery_dir, Pizzaiolo.CSV_DIR)
        csv_name = os.path.join(csv_dir, Pizzaiolo.CSV_FILENAME)
        df = pd.read_csv(csv_name, dtype=str)
        types = df[Pizzaiolo.TYPE_COLUMN_NAME].unique()

        train_df = pd.DataFrame()
        inter_df = pd.DataFrame()
        for t in types:
            train, inter = train_test_split(df[df[Pizzaiolo.TYPE_COLUMN_NAME] == t], train_size=train_size)
            train_df = pd.concat([train_df, train])
            inter_df = pd.concat([inter_df, inter])

        if test_size is None:
            valid_df = inter_df
        else:
            df = inter_df
            valid_df = pd.DataFrame()
            test_df = pd.DataFrame()
            v_size = valid_size * 1 / inter_size
            for t in types:
                valid, test = train_test_split(df[df[Pizzaiolo.TYPE_COLUMN_NAME] == t], train_size=v_size)
                valid_df = pd.concat([valid_df, valid])
                test_df = pd.concat([test_df, test])

        # save files
        train_df.to_csv(os.path.join(csv_dir, Pizzaiolo.CSV_TRAIN_FILENAME), index=False)
        valid_df.to_csv(os.path.join(csv_dir, Pizzaiolo.CSV_VALID_FILENAME), index=False)
        if test_size is not None:
            test_df.to_csv(os.path.join(csv_dir, Pizzaiolo.CSV_TEST_FILENAME), index=False)

    
    @staticmethod
    def __subMasks(mask, other):
        """
        Returns:
            - mask masked with over
        """
        mask = mask.astype(int) - other.astype(int)
        mask[mask<0] = 0
        mask = np.array(mask, dtype='uint8')        
        return mask

    # --------------------------------------------- INSTANCE ------------------------------------------------

    def __init__(self, ontology_filename:str=None):
        """ 
        Creates a Pizzaiolo instance to create a Pizzas dataset
        based on the Pizza ontology.
        
        Parameters:
        - ontology_filename : optional filename of an owlready2.Ontology Pizza ontology
        """

        if (ontology_filename is None):
            self._onto_filename = "libs/ontologies/pizzaiolo.xml"
        else:
            self._onto_filename = ontology_filename
        
        self.__ontology = owlready2.get_ontology(self._onto_filename).load()
        self.__ontology.name = 'pizzaiolo'
        self.pizza = self.__ontology.get_namespace(self.__ontology.get_base_iri())
        self.__init_elements()

    def __init_elements(self):
        self.__elements_dir = "libs/elements/"

        self._elements = {}

        self._elements['base'] = Image.open(self.__elements_dir + "pizza.png")
        self._elements['base'].thumbnail((224, 224))
        
        self._elements['deep_pan'] = Image.open(self.__elements_dir + "pizza_base_dp.png")
        self._elements['deep_pan'].thumbnail((224, 224))
        
        self._elements['thin_crispy'] = Image.open(self.__elements_dir + "pizza_base_tc.png")
        self._elements['thin_crispy'].thumbnail((224, 224))

        self._elements['anchovy'] = Image.open(self.__elements_dir + "anchovy.png")
        self._elements['anchovy'].thumbnail((50,50))

        self._elements['bacon'] = Image.open(self.__elements_dir + "bacon.png")
        self._elements['bacon'].thumbnail((40,40))

        self._elements['garlic'] = Image.open(self.__elements_dir + "garlic.png")
        self._elements['garlic'].thumbnail((15,15))

        self._elements['japaleno'] = Image.open(self.__elements_dir + "japaleno.png")
        self._elements['japaleno'].thumbnail((20,20))

        self._elements['mushroom'] = Image.open(self.__elements_dir + "mushroom.png")
        self._elements['mushroom'].thumbnail((20,20))

        self._elements['olive'] = Image.open(self.__elements_dir + "olive.png")
        self._elements['olive'].thumbnail((15,15))

        self._elements['onion'] = Image.open(self.__elements_dir + "onion.png")
        self._elements['onion'].thumbnail((30,30))

        self._elements['parmesan'] = Image.open(self.__elements_dir + "parmesan.png")
        self._elements['parmesan'].thumbnail((80,80))

        self._elements['pepper_green'] = Image.open(self.__elements_dir + "pepper_green.png")
        self._elements['pepper_green'].thumbnail((35,35))

        self._elements['peperonata'] = Image.open(self.__elements_dir + "peperonata.png")
        self._elements['peperonata'].thumbnail((35,35))

        self._elements['pepperoni'] = Image.open(self.__elements_dir + "pepperoni.png")
        self._elements['pepperoni'].thumbnail((30,30))

        self._elements['prawn'] = Image.open(self.__elements_dir + "prawn.png")
        self._elements['prawn'].thumbnail((50,50))

        self._elements['raisin'] = Image.open(self.__elements_dir + "raisin.png")
        self._elements['raisin'].thumbnail((11,11))

        self._elements['rocket'] = Image.open(self.__elements_dir + "rocket.png")
        self._elements['rocket'].thumbnail((50,50))

        self._elements['spinash'] = Image.open(self.__elements_dir + "spinash.png")
        self._elements['spinash'].thumbnail((35,35))

        self._elements['tomato'] = Image.open(self.__elements_dir + "tomato.png") #, Image.ANTIALIAS)
        self._elements['tomato'].thumbnail((45, 45))
        
        self._elements['america'] = Image.open(self.__elements_dir + "american.png").convert('RGBA')
        self._elements['america'].thumbnail(Pizzaiolo.FLAG_SIZE)
        
        self._elements['england'] = Image.open(self.__elements_dir + "england.png").convert('RGBA')
        self._elements['england'].thumbnail(Pizzaiolo.FLAG_SIZE)
        
        self._elements['france'] = Image.open(self.__elements_dir + "france.png").convert('RGBA')
        self._elements['france'].thumbnail(Pizzaiolo.FLAG_SIZE)
        
        self._elements['italy'] = Image.open(self.__elements_dir + "italy.png").convert('RGBA')
        self._elements['italy'].thumbnail(Pizzaiolo.FLAG_SIZE)
        
        
        self.__toppings = [ # DONT CHANGE THE ORDERING
            self.pizza.ParmesanTopping,
            self.pizza.SlicedTomatoTopping,
            self.pizza.GreenPepperTopping,
            self.pizza.OnionTopping,
            self.pizza.PeperoniSausageTopping,
            self.pizza.HamTopping,
            self.pizza.PeperonataTopping,
            self.pizza.RocketTopping,
            self.pizza.SpinachTopping,
            self.pizza.JalapenoPepperTopping,
            self.pizza.AnchovyTopping,
            self.pizza.MushroomTopping,
            self.pizza.PrawnTopping,
            self.pizza.OliveTopping,
            self.pizza.SultanaTopping,
            self.pizza.GarlicTopping
        ]
        
        
        self.__elements_order = [None]

        if not self.pizza.hasBase is None:
            self.__elements_order.extend([
                self.pizza.DeepPanBase,
                self.pizza.ThinAndCrispyBase
            ])
        
        self.__elements_order.extend(self.__toppings) 
        
        if not self.pizza.hasCountryOfOrigin is None:
            self.__elements_order.extend([
                self.pizza.America,
                self.pizza.England,
                self.pizza.France,
                self.pizza.Italy,
            ])

        self.__order_map = {}
        self.__elements_ids = {}
        self.__ids_elements = {}
        for i, t in enumerate(self.__elements_order):
            self.__order_map[t] = i
            t_str = t.name if not t is None else str(t) 
            self.__elements_ids[t_str] = i
            self.__ids_elements[i] = t_str
            
            
        self._gen_map = {
            self.pizza.AnchovyTopping: self._generateAnchovies,
            self.pizza.GarlicTopping: self._generateGarlic,
            self.pizza.GreenPepperTopping: self._generatePepper,
            self.pizza.HamTopping: self._generateBacon,
            self.pizza.JalapenoPepperTopping: self._generateJapaleno,
            self.pizza.MushroomTopping: self._generateMushroom,
            self.pizza.OliveTopping: self._generateOlive,
            self.pizza.OnionTopping: self._generateOnion,
            self.pizza.ParmesanTopping: self._generateParmesan,
            self.pizza.PeperonataTopping: self._generatePeperonata,
            self.pizza.PeperoniSausageTopping: self._generatePepperoni,
            self.pizza.PrawnTopping: self._generatePrawn,
            self.pizza.RocketTopping: self._generateRocket,
            self.pizza.SlicedTomatoTopping: self._generateSlicedTomato,
            self.pizza.SpinachTopping: self._generateSpinash,
            self.pizza.SultanaTopping: self._generateRaisin,
            
            self.pizza.America: self._pasteAmerica,
            self.pizza.England: self._pasteEngland,
            self.pizza.France: self._pasteFrance,
            self.pizza.Italy: self._pasteItaly,
            
            self.pizza.DeepPanBase: self._pasteDeepPan,
            self.pizza.ThinAndCrispyBase: self._pasteThinCrispy,
        }

    def colorSegmentation(self, segmentation):
        mono_mask = segmentation.squeeze()
        new_shape = segmentation.shape[0], segmentation.shape[1], 3
        colored_mask = np.zeros(new_shape)

        for id in np.unique(segmentation):
            entity_str = self.__ids_elements[id]
            color = Pizzaiolo.getColorFor(entity_str)
            indices = np.where(mono_mask==id)
            colored_mask[indices] = color

        return colored_mask.astype('uint8')

    def cook(self, pizza_types:list, number_of_each=1, delivery_dir='pizza_dataset', pizza_base=None):
        """ Generates a synthetic pizza images dataset
        
        Parameters:
        - pizza_types: a list of ontological pizza types
        - number_of_each: number of pizza images generated for eachy pizza type
        - delivery_dir: directory for the generated dataset (WARNING : will be erased and replaced !!!)
        - pizza_base: the base image over which everything will be generated
        """

        if len(pizza_types) == 0:
            return

        if not os.path.exists(delivery_dir):
            os.mkdir(delivery_dir)
        else:
            raise Exception('Unable to cook the dataset : directory %s already exists !' % delivery_dir)
        
        img_dir = os.path.join(delivery_dir, Pizzaiolo.IMAGES_DIR)
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        labels_dir = os.path.join(delivery_dir, Pizzaiolo.LABELS_DIR)
        if not os.path.exists(labels_dir):
            os.mkdir(labels_dir)
        csv_dir = os.path.join(delivery_dir, Pizzaiolo.CSV_DIR)
        if not os.path.exists(csv_dir):
            os.mkdir(csv_dir)     
        ontology_dir = os.path.join(delivery_dir, Pizzaiolo.ONTOLOGY_DIR)
        if not os.path.exists(ontology_dir):
            os.mkdir(ontology_dir)

        if pizza_base is None:
            pizza_base = self.getDefaultBase()

        possible_toppings = [t.name for t in self.__toppings]
        possible_toppings.sort()

        # Génération des data
        img_idx = 0
        data = []
        ignored_entities = []
        json_indents = 4
        for p in pizza_types:
            for i in range(number_of_each):
                image, entities  = self.preparePizza(p, base_img=pizza_base).values()
                names, all_counts, all_masks, all_contours, all_boxes, not_supported = entities.values()
                
                hasTopping = self.pizza.hasTopping
                hasBase = self.pizza.hasBase
                hasCountryOfOrigin = self.pizza.hasCountryOfOrigin
                
                topping_names = names[hasTopping.name] 
                
                try: base_name = names[hasBase.name] 
                except: base_name = None
                
                try: country_name = names[hasCountryOfOrigin.name] 
                except: country_name = None

                for t in not_supported:
                    if ignored_entities.count(t) <= 0: ignored_entities.append(t)

                ref = "img_%05d" % img_idx

                img_name = ref + Pizzaiolo.IMAGES_SUFFIX
                img_idx += 1
                image.save(os.path.join(img_dir, img_name))

                pizzaiolo_boxes_name = ref + Pizzaiolo.BOXES_SUFFIX
                bboxes_json = json.dumps(all_boxes, indent=json_indents)
                self.__writeJson(os.path.join(labels_dir, pizzaiolo_boxes_name), bboxes_json)

                # save contours
                for t, contours in all_contours.items():
                    for i, c in enumerate(contours):
                        contours[i] = c.tolist()
                pizzaiolo_contours_name = ref + Pizzaiolo.CONTOURS_SUFFIX
                contours_json = json.dumps(all_contours, indent=json_indents)
                self.__writeJson(os.path.join(labels_dir, pizzaiolo_contours_name), contours_json)
                
                # save segmentation
                segmentation = self.getSegmentationFromMasks(image, all_masks)
                pizzaiolo_seg_name = ref + Pizzaiolo.SEGMENTATION_SUFFIX
                np.savetxt(os.path.join(labels_dir, pizzaiolo_seg_name), segmentation, fmt='%u')

                # dataframe row
                if p is None:
                    pname = 'None'
                else:
                    pname = p.name
                newRow = [ref, img_name, pname, topping_names]
                
                # topping instances
                for t in possible_toppings:
                    try:
                        count = all_counts[t]
                    except KeyError:
                        count = 0
                    newRow.append(count)
                
                if not hasBase is None: newRow.append(base_name)
                if not hasCountryOfOrigin is None: newRow.append(country_name)
                newRow.extend([pizzaiolo_boxes_name, pizzaiolo_contours_name, pizzaiolo_seg_name])
                data.append(newRow)

        # entities ids
        seg_ids_name = Pizzaiolo.ENTITIES_FILENAME
        seg_ids_json = json.dumps(self.__ids_elements, indent=json_indents)
        self.__writeJson(os.path.join(labels_dir, seg_ids_name), seg_ids_json)

        # dataframe
        cols = [Pizzaiolo.REF_COLUMN_NAME, 'img_name', Pizzaiolo.TYPE_COLUMN_NAME, hasTopping.name] 
        for t in possible_toppings:
            cols.append(t)
        if not hasBase is None: cols.append(Pizzaiolo.HASBASE_COLUMN_NAME)
        if not hasCountryOfOrigin is None: cols.append(Pizzaiolo.HASCOUNTRY_COLUMN_NAME)
        cols.extend(['boxes_name', 'contours_name', 'segmentation_name'])
        df = pd.DataFrame(data=data, columns=cols)
        df_name = os.path.join(csv_dir, Pizzaiolo.CSV_FILENAME)
        df.to_csv(df_name, index=False)
        
        # ontology
        onto_name = self._onto_filename.split(os.sep)[-1]
        onto_name = os.path.join(ontology_dir, onto_name)
        shutil.copyfile(self._onto_filename, onto_name)
        
        if len(ignored_entities) > 0:
            print("Ignored Toppings : ", ignored_entities)
        

    def __cropImage(self, topping_img):
        """
        Crops the image while removing surrounding transparent pixels
        
        Returns:
        - the cropped image
        """
        arr = np.array(topping_img)

        # annulation de la couche alpha
        arr[:,:,-1] = 0
        arr = arr.sum(axis=(2))

        # recupération des coords
        result = np.where(arr != 0)
        y_min = np.min(result[0])
        y_max = np.max(result[0])
        x_min = np.min(result[1])
        x_max = np.max(result[1])

        # retaillage
        topping_img = topping_img.crop((x_min, y_min, x_max, y_max))

        return topping_img
        
    
    def _generateTopping(self, base_img, topping_img, number=None, limits=(0., 1.), masking=True):
        """
        Returns:
            a PIL Image which is the basse_im completed with number * topping_img
            a list of masks (len = number) containing a binary mask for each generated topping
        """
        
        new_img = Image.new('RGBA', (base_img.size[0], base_img.size[1]), (0,0,0,0))

        if number == None:
            number = randint(1, 10)

        center_x, center_y = new_img.size[0] // 2, new_img.size[1] // 2
        topping_size = max(topping_img.size)
        margin = 5
        radius = min(center_x, center_y) - topping_size//2 - margin

        theta_step = (2 * math.pi) / number

        masks = []
        theta = random() * 2 * math.pi
        for i in range(number):
            if limits[0] != limits[1]:
                r = (randrange( int(radius * limits[0]), int(radius * limits[1])) )
            else:
                r = radius * limits[0]
            theta = theta + theta_step
            
            new_topping = topping_img.copy()
            new_topping = new_topping.rotate(randint(0, 360), expand=True)
            new_topping = self.__cropImage(new_topping)

            topping_w, topping_h = new_topping.size
            x = int(center_x + r * math.cos(theta)) - topping_w//2
            y = int(center_y + r * math.sin(theta)) - topping_h//2

            new_img, mask = self._paste(new_img, new_topping, (x, y), masking)

            if not mask is None:
                for i, m in enumerate(masks):
                    m = m.astype(int) - mask.astype(int)
                    m[m<0] = 0
                    masks[i] = np.array(m, dtype='uint8')

                masks.append(mask)

        # génération de l'image
        full_img, _ = self._paste(base_img, new_img, masking=False)

        return full_img, masks if masking else None

    def _generateAnchovies(self, base_img, nb_min=3, nb_max=5, radius_min=0.8, radius_max=0.8, masking=True):
        """
        Returns:
            - a PIL Image which is the base_im completed with a random number of topping images
            - a list of masks each containing a binary mask for each generated topping
        """
        return self._generateTopping(base_img, self.__getSome('anchovy'),
                                     randint(nb_min, nb_max), 
                                     limits=(radius_min, radius_max), 
                                     masking=masking)

    def _generateBacon(self, base_img, nb_min=3, nb_max=4, radius_min=0.4, radius_max=0.4, masking=True):
        """
        Returns:
            - a PIL Image which is the base_im completed with a random number of topping images
            - a list of masks each containing a binary mask for each generated topping
        """
        return self._generateTopping(base_img, self.__getSome('bacon'),
                                     randint(nb_min, nb_max), 
                                     limits=(radius_min, radius_max), 
                                     masking=masking)

    def _generateGarlic(self, base_img, nb_min=3, nb_max=6, radius_min=0.7, radius_max=0.85, masking=True):
        """
        Returns:
            - a PIL Image which is the base_im completed with a random number of topping images
            - a list of masks each containing a binary mask for each generated topping
        """
        return self._generateTopping(base_img, self.__getSome('garlic'),
                                     randint(nb_min, nb_max), 
                                     limits=(radius_min, radius_max), 
                                     masking=masking)

    def _generateJapaleno(self, base_img, nb_min=2, nb_max=4, radius_min=0.15, radius_max=0.15, masking=True):
        """
        Returns:
            - a PIL Image which is the base_im completed with a random number of topping images
            - a list of masks each containing a binary mask for each generated topping
        """
        return self._generateTopping(base_img, self.__getSome('japaleno'),
                                     randint(nb_min, nb_max), 
                                     limits=(radius_min, radius_max), 
                                     masking=masking)

    def _generateMushroom(self, base_img, nb_min=6, nb_max=10, radius_min=0.7, radius_max=0.85, masking=True):
        """
        Returns:
            - a PIL Image which is the base_im completed with a random number of topping images
            - a list of masks each containing a binary mask for each generated topping
        """
        return self._generateTopping(base_img, self.__getSome('mushroom'),
                                     randint(nb_min, nb_max), 
                                     limits=(radius_min, radius_max), 
                                     masking=masking)

    def _generateOlive(self, base_img, nb_min=6, nb_max=10, radius_min=0.2, radius_max=0.65, masking=True):
        """
        Returns:
            - a PIL Image which is the base_im completed with a random number of topping images,
            - a list of masks each containing a binary mask for each generated topping
        """
        return self._generateTopping(base_img, self.__getSome('olive'),
                                     randint(nb_min, nb_max), 
                                     limits=(radius_min, radius_max), 
                                     masking=masking)

    def _generateOnion(self, base_img, nb_min=3, nb_max=5, radius_min=0.3, radius_max=0.4, masking=True):
        """
        Returns:
            - a PIL Image which is the base_im completed with a random number of topping images,
            - a list of masks each containing a binary mask for each generated topping
        """
        return self._generateTopping(base_img, self.__getSome('onion'),
                                     randint(nb_min, nb_max), 
                                     limits=(radius_min, radius_max), 
                                     masking=masking)

    def _generateParmesan(self, base_img, nb_min=1, nb_max=1, radius_min=0.0, radius_max=0.0, masking=True):
        """
        Returns:
            - a PIL Image which is the base_im completed with a random number of topping images
            - a list of masks each containing a binary mask for each generated topping
        """
        return self._generateTopping(base_img, self.__getSome('parmesan'),
                                     randint(nb_min, nb_max), 
                                     limits=(radius_min, radius_max), 
                                     masking=masking)

    def _generatePepper(self, base_img, nb_min=5, nb_max=7, radius_min=0.7, radius_max=0.8, masking=True):
        """
        Returns:
            - a PIL Image which is the base_im completed with a random number of topping images
            - a list of masks each containing a binary mask for each generated topping
        """
        return self._generateTopping(base_img, self.__getSome('pepper_green'),
                                     randint(nb_min, nb_max), 
                                     limits=(radius_min, radius_max), 
                                     masking=masking)

    def _generatePeperonata(self, base_img, nb_min=1, nb_max=1, radius_min=0.0, radius_max=0.0, masking=True):
        """
        Returns:
            - a PIL Image which is the base_im completed with a random number of topping images
            - a list of masks each containing a binary mask for each generated topping
        """
        return self._generateTopping(base_img, self.__getSome('peperonata'),
                                     randint(nb_min, nb_max), 
                                     limits=(radius_min, radius_max), 
                                     masking=masking)

    def _generatePepperoni(self, base_img, nb_min=5, nb_max=7, radius_min=0.4, radius_max=0.55, masking=True):
        """
        Returns:
            - a PIL Image which is the base_im completed with a random number of topping images
            - a list of masks each containing a binary mask for each generated topping
        """
        return self._generateTopping(base_img, self.__getSome('pepperoni'),
                                     randint(nb_min, nb_max), 
                                     limits=(radius_min, radius_max), 
                                     masking=masking)

    def _generatePrawn(self, base_img, nb_min=5, nb_max=8, radius_min=0.7, radius_max=0.8, masking=True):
        """
        Returns:
            - a PIL Image which is the base_im completed with a random number of topping images,
            - a list of masks each containing a binary mask for each generated topping
        """
        return self._generateTopping(base_img, self.__getSome('prawn'),
                                     randint(nb_min, nb_max), 
                                     limits=(radius_min, radius_max), 
                                     masking=masking)

    def _generateRaisin(self, base_img, nb_min=6, nb_max=10, radius_min=0.5, radius_max=0.85, masking=True):
        """
        Returns:
            - a PIL Image which is the base_im completed with a random number of topping images
            - a list of masks each containing a binary mask for each generated topping
        """
        return self._generateTopping(base_img, self.__getSome('raisin'),
                                     randint(nb_min, nb_max), 
                                     limits=(radius_min, radius_max), 
                                     masking=masking)

    def _generateRocket(self, base_img, nb_min=2, nb_max=4, radius_min=0.4, radius_max=0.6, masking=True):
        """
        Returns:
            - a PIL Image which is the base_im completed with a random number of topping images
            - a list of masks each containing a binary mask for each generated topping
        """
        return self._generateTopping(base_img, self.__getSome('rocket'),
                                     randint(nb_min, nb_max), 
                                     limits=(radius_min, radius_max), 
                                     masking=masking)

    def _generateSlicedTomato(self, base_img, nb_min=4, nb_max=6, radius_min=0.55, radius_max=0.75, masking=True):
        """
        Returns:
            - a PIL Image which is the base_im completed with a random number of topping images
            - a list of masks each containing a binary mask for each generated topping
        """
        return self._generateTopping(base_img, self.__getSome('tomato'),
                                     randint(nb_min, nb_max), 
                                     limits=(radius_min, radius_max), 
                                     masking=masking)

    def _generateSpinash(self, base_img, nb_min=2, nb_max=4, radius_min=0.3, radius_max=0.45, masking=True):
        """
        Returns:
            - a PIL Image which is the base_im completed with a random number of topping images
            - a list of masks each containing a binary mask for each generated topping
        """
        return self._generateTopping(base_img, self.__getSome('spinash'), 
                                     randint(nb_min, nb_max), 
                                     limits=(radius_min, radius_max), 
                                     masking=masking)

    def getDefaultBase(self):
        """
        Returns:
            - a PIL Image of the default pizza base
        """
        return self.__getSome('base')

    def _getElementsIds(self):
        return self.__elements_ids

    def _getElementFromId(self, id):
        return self.__ids_elements[id]

    def __getIdFromElement(self, topping_str):
        return self.__elements_ids[topping_str]

    def getOntology(self)->owlready2.Ontology:
        """
        Returns:
        - the owlready2.Ontology instance used to generate pizzas
        """
        return self.__ontology

    def getRandomBase(self):
        """
        Returns:
            - a PIL Image of a random pizza base made of ramdom pixels
        """
        SIZE = self.__getSome('base').size[0]
        radius = SIZE * randrange(95, 100) / 100.0 #0.95
        coords = (SIZE - radius)
        color = 'yellow'

        ellipse = Image.new('RGB', (SIZE, SIZE))
        draw = ImageDraw.Draw(ellipse)
        draw.ellipse((coords, coords, radius, radius), fill=color, outline =color)
        imrand = np.random.rand(SIZE, SIZE, 3) * 255
        base = np.multiply(ellipse, imrand)
        base = Image.fromarray(base.astype('uint8')).convert('RGBA')
        return base

    def __getSome(self, element):
        try:
            return self._elements[element].copy()
        except KeyError:
            print("No such element : should be one of %s" % [name for name in self._elements.keys()],
                  file=sys.stderr)

    def getSegmentationFromMasks(self, img, masks):
        final_mask = np.zeros(img.size, dtype='uint8')
        for entity, mask in masks.items():
            id = self.__getIdFromElement(entity)
            if type(mask) is list:
                for sub_mask in mask:
                    final_mask[sub_mask>0] = id
            else:
                final_mask[mask>0] = id
        return final_mask

    def getSegmentationFromContours(self, img, contours):
        mask = np.zeros(img.size, dtype='uint8')
        for t, contours in contours.items():
            id = self.__getIdFromElement(t)
            cv2.drawContours(
                image=mask,
                contours=contours,
                contourIdx=-1,
                color=id,
                thickness=cv2.FILLED)
        return mask
    
    def _paste(self, base_img, el_img, coords=(0,0), masking=True):
        """
        Duplicates base_img and pastes el_im.
        
        Returns:
        - a copy of the base_img pasted with el_im
        - the mask of the pasted element
        """
        new_img = base_img.copy()
        new_img.paste(el_img, (coords[0], coords[1]), el_img)
        
        if masking:
            mask = Image.new('RGBA', (base_img.size[0], base_img.size[1]), (0,0,0,0))
            mask.paste(el_img, (coords[0], coords[1]), el_img)
            mask = np.array(mask.split()[3], dtype='uint8')
            mask[mask>0] = 1
            
        return new_img, mask if masking else None

    def _pasteFlag(self, base_img, flag_img, masking=True):
        """
        Duplicates base_img and pastes flag_img.
        
        Returns:
        - a copy of the base_img pasted with flag_img
        - the mask of the pasted element
        """
        x_c = (int) (base_img.size[0]/2) 
        y_c = (int) (base_img.size[1]/2) 
        rayon = x_c - flag_img.size[0]/2 + 10
        angle = - math.pi / 4
        x = x_c + (int)(rayon * math.cos(angle)) 
        y = y_c + (int)(rayon * math.sin(angle)) 
        return self._paste(base_img, flag_img, (x,y))
    
    def _pasteAmerica(self, base_img, masking=True):
        """
        Duplicates base_img and pastes a flag.
        
        Returns:
        - a copy of the pasted base_img
        - the mask of the pasted element
        """
        return self._pasteFlag(base_img, self.__getSome('america'))
    
    def _pasteEngland(self, base_img, masking=True):
        """
        Duplicates base_img and pastes a flag.
        
        Returns:
        - a copy of the pasted base_img
        - the mask of the pasted element
        """
        return self._pasteFlag(base_img, self.__getSome('england'))
    
    def _pasteFrance(self, base_img, masking=True):
        """
        Duplicates base_img and pastes a flag.
        
        Returns:
        - a copy of the pasted base_img
        - the mask of the pasted element
        """
        return self._pasteFlag(base_img, self.__getSome('france'))
    
    def _pasteItaly(self, base_img, masking=True):
        """
        Duplicates base_img and pastes a flag.
        
        Returns:
        - a copy of the pasted base_img
        - the mask of the pasted element
        """
        return self._pasteFlag(base_img, self.__getSome('italy'))
    
    def _pasteThinCrispy(self, base_img, masking=True):
        """
        Duplicates base_img and pastes a base.
        
        Returns:
        - a copy of the pasted base_img
        - the mask of the pasted element
        """
        return self._paste(base_img, self.__getSome('thin_crispy'))

    def _pasteDeepPan(self, base_img, masking=True):
        """
        Duplicates base_img and pastes a base.
        
        Returns:
        - a copy of the pasted base_img
        - the mask of the pasted element
        """
        return self._paste(base_img, self.__getSome('deep_pan'))

    def preparePizza(self, pizza_type, base_img=None, background=None):
        """
        Generates a pizza image + informations corresponding to the ontological pizza_type
        
        Parameters:
        - pizza_type: an ontological named pizza type (or None for anonymous random pizza)
        - base: the base image on which are generated all entities (toppings, etc.)
        - background: background image on which the base is pasted
        """
        if base_img == None:
            base_img = self.getDefaultBase()
        if background != None:
            pizza_img = background.copy()
            base_img = base_img.resize((200, 200))
            shift = int((pizza_img.size[0] - base_img.size[0]) / 2)
            pizza_img.paste(base_img, (shift, shift), base_img)
        else:
            pizza_img = base_img
            
        all_counts = {}
        all_masks = {}
        all_contours = {}
        all_boxes = {}
        not_supported = []
        
        # hasBase
        base_name = None
        try:
            if pizza_type is None:
                possible_bases = list(self.pizza.PizzaBase.descendants())
                possible_bases.remove(self.pizza.PizzaBase)
                base = choice(possible_bases)
            else:
                base = pizza_type.hasBase
                
            if not base is None:
                base_name = base.name 
                try:
                    pizza_img, base_mask = self._gen_map[base](pizza_img)
                    all_masks[base_name] = base_mask
                except Exception as ex:
                    not_supported.append(base)
        except:
            # hasBase n'existe pas
            pass
                
        
        # hasTopping
        toppings = {}
        all_toppings_masks = {}
        
        if pizza_type is None:
            has_topping = set(choices(self.__toppings, k=randrange(1,6)))
        else:
            has_topping = pizza_type.hasTopping
            
        for t in has_topping:
            try:
                toppings[self.__order_map[t]] = t
            except:
                not_supported.append(t)

        for k in sorted(toppings):
            topping = toppings[k]
            pizza_img, topping_masks = self._gen_map[topping](pizza_img)

            full_topping_mask = np.sum(topping_masks, axis=0)
            full_topping_mask[full_topping_mask>0] = 1

            # masquage entre toppings 
            for t, masks in all_toppings_masks.items():
                for i, m in enumerate(masks):
                    masks[i] = Pizzaiolo.__subMasks(m, full_topping_mask)                    

            all_toppings_masks[topping.name] = topping_masks
        
        for t, masks in all_toppings_masks.items():
            all_masks[t] = all_toppings_masks[t]
            all_contours[t] = []
            all_boxes[t] = []
            for i, m in enumerate(masks):
                contours, hierarchy = Pizzaiolo._findMaskContours(m)
                all_contours[t].extend(contours)
                boxes = Pizzaiolo._findBoxes(contours, hierarchy)
                all_boxes[t].extend(boxes)
                  
        
        # hasCountryOfOrigin
        country_name = None
        try:
            if pizza_type is None:
                possible_countries = [
                    self.pizza.America,
                    self.pizza.England,
                    self.pizza.France,
                    self.pizza.Italy,
                    None
                ]
                country = choice(possible_countries)
            else:
                country = pizza_type.hasCountryOfOrigin
                
            if not country is None:
                country_name = country.name
                try:
                    pizza_img, country_mask = self._gen_map[country](pizza_img)
                    
                    # mask with toppings
                    all_masks[country_name] = Pizzaiolo.__subMasks(country_mask, full_topping_mask)
                    
                    country_contour, country_hierarchy = Pizzaiolo._findMaskContours(country_mask)
                    all_contours[country_name] = country_contour
                    country_box = Pizzaiolo._findBoxes(country_contour, country_hierarchy)
                    all_boxes[country_name] = country_box
                    all_counts[country_name] = 1
                except:
                    not_supported.append(country)
        except:
            # hasCountryOfOrigin n'existe pas
            pass

        # hasBase : masquage
        if not base_name is None:
            
            # mask withcountry
            if not country_name is None:
                all_masks[base_name] = Pizzaiolo.__subMasks(all_masks[base_name], all_masks[country_name])
            
            # mask with toppings 
            all_masks[base_name] = Pizzaiolo.__subMasks(all_masks[base_name], full_topping_mask)
            
            base_contour, base_hierarchy = Pizzaiolo._findMaskContours(all_masks[base_name])
            all_contours[base_name] = base_contour
            
            all_counts[base_name] = 1
            base_boxes = Pizzaiolo._findBoxes(base_contour, base_hierarchy)
            all_boxes[base_name] = base_boxes
            
        # counts after clippings
        for t in has_topping:
            try:
                count = len(all_boxes[t.name])
                all_counts[t.name] = count
            except KeyError:
                raise PizzaGenerationException("%s generation error : the number of %s after clipping is null" % (pizza_type.name, t.name))

        # overlapping rate between toppings to evaluate pizza "readability"
        overlapping_rate = self._get_overlapping_rate(has_topping, all_boxes)
        if overlapping_rate > 0.40:
            return self.preparePizza(pizza_type, base_img, background)

        # results
        result = {
            'image': pizza_img,
            'entities': {
                'names':{
                    self.__ontology.hasTopping.name: [t.name for t in toppings.values()],
                },
                'counts': all_counts,
                'masks': all_masks,
                'contours': all_contours,
                'boxes': all_boxes,
                'not_supported': not_supported
            }
        }
        
        if not base_name is None: result['entities']['names'][self.__ontology.hasBase.name] = base_name
        if not country_name is None: result['entities']['names'][self.__ontology.hasCountryOfOrigin.name] = country_name
            
        return result

    def __writeJson(self, filename, json_object):
        with open(filename, "w") as outfile:
            outfile.write(json_object)
    
    def __get_intersection(self, bbox1, bbox2):
        """
        Calculate the Intersection of two bounding boxes.
        """
        # determine the coordinates of the intersection rectangle
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[0]+bbox1[2], bbox2[0]+bbox2[2])
        y_bottom = min(bbox1[1]+bbox1[3], bbox2[1]+bbox2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0
        return (x_right - x_left) * (y_bottom - y_top)


    def _get_overlapping_rate(self, has_topping, all_boxes):
        """
        For each topping, calculates the surface of all its boxes and their intersection with other toppings
        Then calculates the ratio intersection/surface for each topping
        And returns the higher ratio. 
        """
        overlapping_rates = []
        excluded_toppings = [self.pizza.OliveTopping, 
                             self.pizza.ParmesanTopping,
                             self.pizza.PeperonataTopping, 
                             self.pizza.JalapenoPepperTopping]
        toppings = [t for t in has_topping if t not in excluded_toppings]
        for t in toppings:
            if t in excluded_toppings:
                continue
            t_surface = 0.0001
            t_overlap = 0.0
            toppings.remove(t)
            other_boxes = sum([all_boxes[tt.name] for tt in has_topping if tt not in sum([[t], excluded_toppings], [])], [])
            for box in all_boxes[t.name]:
                t_surface += box[2] * box[3]
                for other_box in other_boxes:
                    t_overlap += self.__get_intersection(box, other_box)
            if t_overlap / t_surface > 1.0:
                overlapping_rates.append(1.0)
            else:
                overlapping_rates.append(t_overlap / t_surface)
        if len(overlapping_rates) == 0 :
            overlapping_rates.append(0.0)
        return max(overlapping_rates)
        

class PizzaGenerationException(Exception):
    def __init__(self, msg) -> None:
        return super(PizzaGenerationException, self).__init__(msg)