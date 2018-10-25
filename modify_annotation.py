import yaml
import argparse
import os
from yolo3.utils import get_classes, translate_classes

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--config_path",
                required=True,
                default=None,
                type=str,
                help="The training configuration.")
ap.add_argument("-s", "--output_suffix",
                required=True,
                default=None,
                type=str,
                help="The suffix to be appended in the output annotation file.")
ap.add_argument("-t", "--run_class_translations",
                required=False,
                action="store_true",
                help="Run class translations.")
ARGS = ap.parse_args()

train_config = None
with open(ARGS.config_path, 'r') as stream:
    train_config = yaml.load(stream)

class_names = get_classes(train_config['classes_path'])

for annotation_path in [train_config['train_path'], train_config['test_path']]:
    print('Checking annotation',annotation_path)
    did_modify_something = False
    output_annotation_path = annotation_path.replace('.txt', '_{}.txt'.format(ARGS.output_suffix))

    if os.path.exists(output_annotation_path):
        raise Exception('The output file already exists:',output_annotation_path)

    with open(annotation_path) as f:
        lines = f.readlines()

    if ARGS.run_class_translations:
        if 'class_translation_path' in train_config:
            class_translation_config = None
            with open(train_config['class_translation_path'], 'r') as stream:
                class_translation_config = yaml.load(stream)
                lines = translate_classes(lines,class_names,class_translation_config)
                did_modify_something = True
        else:
            raise Exception('The configuration for class translation is not specified in the config file.')

    if did_modify_something:
        with open(output_annotation_path, 'w') as output_f:
            print('Writting the modified annotation file to', output_annotation_path)
            for annot_line in lines:
                output_f.write(annot_line + '\n')
    else:
        print('You did not choose any modification to be applied')

print('Done! Now replace the train and test dataset paths with the new generated ones.')
