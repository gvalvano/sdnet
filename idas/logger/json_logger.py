import json
import io

data = {}


def add_new_node(key, values, fname='log_file.json'):
    """
    Add new node to JSON file under the key=key and with sub-keys=values.
    :param key: key to address the node
    :param values: dictionary with key-values couples
    :param fname: JSON file name to write

    Example:
        data.update({'SPARSE_TRAINING': {'done_before': False, 'beta': 0.10, 'sparsity': 0.30}})
    """
    data.update({key: values})

    with io.open(fname, 'w', encoding='utf8') as outfile:
        str_ = json.dumps(data,
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(str_)


def read_one_node(key, fname='log_file.json'):
    """
    Return the dictionary in JSON file under the key=key.
    :param key: dictionary key
    :param fname: JSON file name to read
    :return: dictionary
    """
    with open(fname, 'r', encoding='utf8') as infile:
        node = json.load(infile)
    return node[key]


def update_node(key, sub_key, sub_value, fname='log_file.json'):
    """
    Update a node in a JSON file under the key=key and with sub-keys=values.
    :param key: key to address the node
    :param sub_key: field name to be updated under the node key
    :param sub_value: value to assign to the field name under the node key
    :param fname: JSON file name to write
    """
    content_dict = read_one_node(key, fname=fname)
    content_dict[sub_key] = sub_value

    data.update({key: content_dict})

    with io.open(fname, 'w', encoding='utf8') as outfile:
        str_ = json.dumps(data,
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(str_)


if __name__ == '__main__':
    # Example of reading and writing on JSON file
    k = 'KEY_1'
    val = {'flag': True, 'alpha': 0.10, 'beta': 0.30}
    add_new_node(k, val)

    k = 'KEY_2'
    val = {'flag': False, 'alpha': 0.20, 'beta': 0.60}
    add_new_node(k, val)

    k = 'KEY_1'
    print(read_one_node(k))
