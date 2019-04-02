import random
import json,os
annotations_file = 'annotations.json'
data_dir = 'fashion_data'
if __name__ == '__main__':
    annots = json.load(open(annotations_file))
    class_distribution = {}
    for annot in annots:
        if annot["class"] not in class_distribution:
            class_distribution[annot["class"]]=[]
        class_distribution[annot["class"]].append((annot["image_path"],annot["class"]))

    class_set = set(class_distribution.keys())
    true_num_classes = len(class_set)
    missing_classes = list(set(range(max(class_set))) - class_set)
    print(missing_classes)
    new_class_map = {}
    for i in range( max(class_set), len(class_set),-1):
        new_class_map[i] = missing_classes.pop()
    train_annots, val_annots = [],[]
    for k,v in class_distribution.items():
        train = random.sample(v, int(0.8*len(v)) )
        val = list(set(v) - set(train) )
        if k in new_class_map:
            k = new_class_map[k]
        train = [[f[0], k] for f in train]
        val = [[f[0],k] for f in val]
        train_annots.extend(train)
        val_annots.extend(val)
    json.dump(train_annots, open(os.path.join(data_dir,"train.json"),"w"),indent=4)
    json.dump(val_annots, open(os.path.join(data_dir,"val.json"), "w"), indent=4)