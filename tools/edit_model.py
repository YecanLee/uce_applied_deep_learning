import os.path as osp
import time
from argparse import ArgumentParser, Namespace

from uce.edit import EDITORS, load_editor, parse_concepts, save_editor
from mmengine import Config, DictAction, mkdir_or_exist
from uce.utils import setup_logger


def parse_args() -> Namespace:
    parser = ArgumentParser('Edit model weights.')
    parser.add_argument('config', help='Path to the config file.')
    parser.add_argument(
        'edit_concepts',
        help="Concepts to edit. You can enter multiple concepts, separated by ','.  "
        "If an argument has format of 'xxx.txt', then concepts are read from "
        'a txt file, where each line is a concept. ')
    parser.add_argument(
        '--guided-concepts',
        '-g',
        help='Target concepts of the editing. You can enter multiple concepts, '
        "separated by ','. If an argument has format of 'xxx.txt', then concepts "
        'are read from a txt file, where each line is a concept. Note that the '
        'number of guided concepts must be equal to the number of edit concepts, '
        'or be 1, which means that all the editing concepts will be guided '
        'towards this single target concept.')
    parser.add_argument(
        '--preserve-concepts',
        '-p',
        help='Concepts to preserve. You can enter multiple concepts, '
        "separated by ','. If an argument has format of 'xxx.txt', "
        'then concepts are read from a txt file, where '
        'each line is a concept. ')
    parser.add_argument(
        '--np',
        type=int,
        help='Number of preserve concepts. If it is given, then randomly sample '
        'this number of concepts from the preserve concepts.')
    parser.add_argument(
        '--with-extra',
        choices=['artist', 'object'],
        help='Type of the extra prompts. If it is artist, then some extra prompts '
        "such as 'painting by {concept}' will be added to the set of "
        "edit concepts. If it is 'object', then extra prompts are like "
        "'image of {concept}'. Otherwise, no extra concepts will be used.")
    parser.add_argument(
        '--work-dir',
        '-w',
        default='./workdirs/default/',
        help='Directory to save the output files.')
    parser.add_argument(
        '--editor-ckpt',
        '-c',
        help='Checkpoint of an editor. If it is given, then it means continue to '
        'edit the given model weight, otherwise edit the pre-trained model '
        'weight. ')
    parser.add_argument(
        '--meta-info',
        '-m',
        help='Path to the meta info json file of the editor. Only needed if you '
        'wanna load a previously used editor and continue editing the model.')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='key value pairs to override the config file entries. The pairs need to '
        'be in format of xxx=yyy')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    work_dir = args.work_dir
    mkdir_or_exist(work_dir)
    device = f'cuda:{args.gpu_id}'

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log_file = osp.join(work_dir, f'{timestamp}.log')
    logger = setup_logger('uce', filepath=log_file)

    cfg = Config.fromfile(args.config)
    cfg.editor.update({'device': device})
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    logger.info(f"Using config:\n{'=' * 60}\n{cfg.pretty_text}\n{'=' * 60}\n")

    editor = EDITORS.build(cfg.editor)
    if args.editor_ckpt is not None:
        editor = load_editor(editor, args.editor_ckpt, args.meta_info).to(device)

    edit_concepts, guided_concepts, preserve_concepts = parse_concepts(
        args.edit_concepts,
        guided_concepts=args.guided_concepts,
        preserve_concepts=args.preserve_concepts,
        num_preserve_concepts=args.np,
        with_extra=args.with_extra,
    )
    editor.edit(edit_concepts, guided_concepts, preserve_concepts)
    save_editor(editor, osp.join(work_dir, f'editor_{editor.id_code}.pt'))


if __name__ == '__main__':
    main()
