import logging
import tensorflow as tf
from google.protobuf import text_format
from protos import string_int_label_map_pb2


def _validate_label_map(label_map):
    for item in label_map.item:
        if item.id < 1:
            raise ValueError('Label map ids should be >= 1.')


def create_category_index(categories):
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index


def convert_label_map_to_categories_person(label_map,
                                    max_num_classes,
                                    use_display_name=True):
    categories_person = []
    list_of_ids_already_added = []
    if not label_map:
        label_id_offset = 1
        for class_id in range(max_num_classes):
            categories_person.append({
                'id': class_id + label_id_offset,
                'name': 'category_{}'.format(class_id + label_id_offset)
            })
        return categories_person
    for item in label_map.item:
        if not 0 < item.id <= max_num_classes:
            logging.info('Ignore item %d since it falls outside of requested '
                         'label range.', item.id)
            continue
        if use_display_name and item.HasField('display_name'):
            name = item.display_name
        else:
            name = item.name
        if item.id not in list_of_ids_already_added:
            list_of_ids_already_added.append(item.id)
            categories_person.append({'id': item.id, 'name': name})
    return categories_person

def convert_label_map_to_categories_mask(label_map,
                                    max_num_classes,
                                    use_display_name=True):
    categories_mask = []
    list_of_ids_already_added = []
    if not label_map:
        label_id_offset = 1
        for class_id in range(max_num_classes):
            categories_mask.append({
                'id': class_id + label_id_offset,
                'name': 'category_{}'.format(class_id + label_id_offset)
            })
        return categories_mask
    for item in label_map.item:
        if not 0 < item.id <= max_num_classes:
            logging.info('Ignore item %d since it falls outside of requested '
                         'label range.', item.id)
            continue
        if use_display_name and item.HasField('display_name'):
            name = item.display_name
        else:
            name = item.name
        if item.id not in list_of_ids_already_added:
            list_of_ids_already_added.append(item.id)
            categories_mask.append({'id': item.id, 'name': name})
    return categories_mask

def load_labelmap_person(path):
    with tf.gfile.GFile(path, 'r') as fid:
        label_map_string = fid.read()
        label_map_person = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map_person)
        except text_format.ParseError:
            label_map_person.ParseFromString(label_map_string)
    _validate_label_map(label_map_person)
    return label_map_person

def load_labelmap_mask(path):
    with tf.gfile.GFile(path, 'r') as fid:
        label_map_string = fid.read()
        label_map_mask = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map_mask)
        except text_format.ParseError:
            label_map_mask.ParseFromString(label_map_string)
    _validate_label_map(label_map_mask)
    return label_map_mask

def get_label_map_dict_person(label_map_path):
    label_map_person = load_labelmap_person(label_map_path)
    label_map_dict_person = {}
    for item in label_map_person.item:
        label_map_dict_person[item.name] = item.id
    return label_map_dict_person

def get_label_map_dict_mask(label_map_path):
    label_map_mask = load_labelmap_mask(label_map_path)
    label_map_dict_mask = {}
    for item in label_map_mask.item:
        label_map_dict_mask[item.name] = item.id
    return label_map_dict_mask
