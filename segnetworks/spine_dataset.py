import os
import pickle
import settings
import numpy as np
from PIL import Image
from datapipe import seg_data


class SpineAccessor(seg_data.SegAccessor):
    def __len__(self):
        return len(self.ds.x_names)

    def get_image_pil(self, sample_i):
        return self.ds.get_pil_image(self.ds.x_names[sample_i])

    def get_labels_arr(self, sample_i):
        pil_img = self.ds.get_pil_image(self.ds.y_names[sample_i])
        y = (np.array(pil_img) / 255).astype(np.int32)
        return y


def _get_spine_path(exists=False):
    return settings.get_data_path(
        config_name='spine',
        dnnlib_template=os.path.join('<DATASETS>', 'research', 'gfrench', 'spine',
                                     'spine_segmentation.zip'),
        exists=exists
    )


class SpineDataSource(seg_data.ZipDataSource):
    def __init__(self):
        super(SpineDataSource, self).__init__(
            _get_spine_path(exists=True))

        sample_names = set()
        # 'train/ISIC_0000000_x.png'
        for filename in self.zip_file.namelist():
            # ('train/ISIC_0000000_x', '.png')
            x_name, ext = os.path.splitext(filename)
            if x_name.endswith('_x') and ext.lower() == '.png':
                sample_name = x_name[:-2]
                sample_names.add(sample_name)
        # sample_name: 'train/ISIC_0000000'
        sample_names = list(sample_names)
        sample_names.sort()
        self.x_names = ['{}_x.png'.format(name) for name in sample_names]
        self.y_names = ['{}_y.png'.format(name) for name in sample_names]
        self.sample_names = sample_names
        # every number respresent a sample case
        self.train_unsup_ndx = np.array([i for i in range(len(self.sample_names))
                                         if self.sample_names[i].startswith('train/')])
        self.train_sup_ndx = np.array([i for i in range(len(self.sample_names))
                                       if self.sample_names[i].startswith('train/FakeMR')])
        self.val_ndx = np.array([i for i in range(len(self.sample_names))
                                 if self.sample_names[i].startswith('val/')])

        self.test_ndx = None

        self.class_names = ['background', 'L1', 'L2', 'L3', 'L4', 'L5']

        self.num_classes = len(self.class_names)

        mean_std = pickle.loads(
            self._read_file_from_zip_as_bytes('rgb_mean_std.pkl'))
        self.rgb_mean = mean_std['rgb_mean']
        self.rgb_std = mean_std['rgb_std']

    def dataset(self, labels, mask, xf, transforms=None, pipeline_type='cv', include_indices=False):
        return SpineAccessor(self, labels, mask, xf, transforms=transforms, pipeline_type=pipeline_type,
                             include_indices=include_indices)

    def get_mean_std(self):
        # For now:
        return (self.rgb_mean, self.rgb_std)
