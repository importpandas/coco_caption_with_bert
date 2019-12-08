from utils import create_input_files
import argparse

if __name__ == '__main__':
    # Create input files (along with word map)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='coco', type=str,
                        help="caption datasets type,i.e.,'coco', 'flickr8k', 'flickr30k' ")
    parser.add_argument("--karpathy_json_path", default='data/dataset_coco.json', type=str,required=True,
                        help="annotation json file' ")
    parser.add_argument("--image_folder", default='data/', type=str,required=True,
                        help="the directory containing the train2014 and val2014 image folders")
    parser.add_argument("--output_folder", default='data/', type=str,required=True,
                        help="path for saving processed data")
    parser.add_argument("--captions_per_image", default=5, type=int)
    parser.add_argument("--min_word_freq", default=5, type=int)
    parser.add_argument("--max_len", default=50, type=int)
    args = parser.parse_args()

    create_input_files(dataset=args.dataset,
                       karpathy_json_path=args.karpathy_json_path,
                       image_folder=args.image_folder,
                       captions_per_image=args.captions_per_image,
                       min_word_freq=args.min_word_freq,
                       output_folder=args.output_folder,
                       max_len=args.max_len)
