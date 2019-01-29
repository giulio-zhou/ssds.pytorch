import cPickle as pickle
import glob
import numpy as np
import os
import pandas as pd
import skimage.io as skio

import json

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from lib.utils.pycocotools.coco import COCO
from lib.utils.pycocotools.cocoeval import COCOeval

class DistillDetection(data.Dataset):
    def __init__(self, root, image_sets, preproc=None, target_transform=None,
                 dataset_name='Distill'):
        self.root = root
        self.ids = list()
        self.annotations = list()
        self.preproc = preproc
        self.target_transform = target_transform

        # For now, assume a single label path.
        label_path = image_sets[0][0]
        self.data = sorted(glob.glob(root + '/*/*'))
        self.labels = []
        label_df = pd.read_csv(label_path)
        self.video_dirs = sorted(glob.glob(root + '/*'))
        self.img_paths = map(lambda x: sorted(glob.glob(x + '/*')),
                             self.video_dirs)
        for video_dir, video_img_paths in zip(self.video_dirs, self.img_paths):
            video_df = label_df[label_df['filename'] == video_dir]
            for frame_no in range(len(video_img_paths)):
                frame_gt = video_df[video_df['frame_no'] == frame_no]
                label_list = []
                if len(frame_gt) > 0:
                    label_list = np.array(frame_gt[['xmin', 'ymin', 'xmax', 'ymax']]).reshape(-1, 4)
                    label_list = map(lambda x: list(x) + [1], label_list)
                self.labels.append(label_list)

        # Set up with standard vals.
        self._classes = ('__background__', 'vehicle')
        self.num_classes = len(self._classes)
        self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
        self._class_to_coco_cat_id = self._class_to_ind

        annofile = self._get_ann_file(label_path)
        self._COCO = COCO(annofile)

    def __getitem__(self, index):
        img = skio.imread(self.data[index])
        height, width = img.shape[:2]
        target = np.array(self.labels[index]).reshape(-1, 5)
        target[:, :4] *= [width, height, width, height]

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return img, target

    def __len__(self):
        return len(self.data)

    def pull_image(self, index):
        img = skio.imread(self.data[index])
        return img

    def pull_anno(self, index):
        img = skio.imread(self.data[index])
        height, width = img.shape[:2]
        target = np.array(self.labels[index]).reshape(-1, 5)
        target[:, :4] *= [width, height, width, height]
        return [index, [(1, t) for t in target]]

    def _get_ann_file(self, label_path, default_path='distill_anno.json'):
        if os.path.exists(default_path):
            return default_path
        df = pd.read_csv(label_path)
        filenames = sorted(np.unique(df['filename']))
        i = 0
        json_file = {}
        json_file['categories'] = [dict(id=1, name='obj', supercategory='obj')]
        json_file['images'], json_file['annotations'] = [], []
        for filename in filenames:
            print(filename)
            img_paths = sorted(glob.glob(filename + '/*'))
            height, width = skio.imread(img_paths[0]).shape[:2]
            for frame_no in range(len(img_paths)):
                frame_df = df[(df['filename'] == filename) &
                              (df['frame_no'] == frame_no)]
                json_file['images'].extend(
                  [{'id': i,
                    'width': width,
                    'height': height,
                    'file_name': img_paths[frame_no]}])
                if len(frame_df) == 0:
                    i += 1
                    continue
                dets = np.array(frame_df[['xmin', 'ymin', 'xmax', 'ymax']])
                dets *= [width, height, width, height]
                xs = dets[:, 0]
                ys = dets[:, 1]
                ws = dets[:, 2] - xs # + 1
                hs = dets[:, 3] - ys # + 1
                xs, ys, ws, hs = map(lambda x: map(float, x), [xs, ys, ws, hs])
                json_file['annotations'].extend(
                  [{'id': i, 'image_id' : i,
                    'category_id' : 1, # only one category
                    'bbox' : [xs[k], ys[k], ws[k], hs[k]],
                    'area': ws[k]*hs[k],
                    'iscrowd': 0} for k in range(dets.shape[0])])
                i += 1
        with open(default_path, 'w') as outfile:
            json.dump(json_file, outfile)
            return default_path

    def _print_detection_eval_metrics(self, coco_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95
        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
               '~~~~'.format(IoU_lo_thresh, IoU_hi_thresh))
        print('{:.1f}'.format(100 * ap_default))
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            print('{:.1f}'.format(100 * ap))

        print('~~~~ Summary metrics ~~~~')
        coco_eval.summarize()

    def _do_detection_eval(self, res_file, output_dir):
        ann_type = 'bbox'
        coco_dt = self._COCO.loadRes(res_file)
        coco_eval = COCOeval(self._COCO, coco_dt)
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        self._print_detection_eval_metrics(coco_eval)
        eval_file = os.path.join(output_dir, 'detection_results.pkl')
        with open(eval_file, 'wb') as fid:
            pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote COCO eval results to: {}'.format(eval_file))

    def _write_coco_results_file(self, all_boxes, res_file):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                          self.num_classes ))
            coco_cat_id = self._class_to_coco_cat_id[cls]
            results.extend(self._coco_results_one_category(all_boxes[cls_ind],
                                                           coco_cat_id))
            '''
            if cls_ind ==30:
                res_f = res_file+ '_1.json'
                print('Writing results json to {}'.format(res_f))
                with open(res_f, 'w') as fid:
                    json.dump(results, fid)
                results = []
            '''
        #res_f2 = res_file+'_2.json'
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def _coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind in range(len(self.data)):
            dets = boxes[im_ind]
            if dets == []:
                continue
            dets = np.float32(dets)
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs # + 1
            hs = dets[:, 3] - ys # + 1
            xs, ys, ws, hs, scores = map(lambda x: map(float, x),
                                         [xs, ys, ws, hs, scores])
            results.extend(
                [{'image_id' : im_ind,
                  'category_id' : cat_id,
                  'bbox' : [xs[k], ys[k], ws[k], hs[k]],
                  'score' : scores[k]} for k in range(dets.shape[0])])
        return results

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = os.path.join(output_dir, ('detection_results'))
        res_file += '.json'
        self._write_coco_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        # if self.coco_name.find('test') == -1:
        self._do_detection_eval(res_file, output_dir)
