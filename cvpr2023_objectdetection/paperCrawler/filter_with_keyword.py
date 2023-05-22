from utils import load_paper_info, filter_papers
import pandas as pd

src_file = 'cvpr2023.csv'

keywords = [
    'object detection',
    'instance segmentation',
    'panoptic segmentation',
    'open-vocabulary',
    'open vocabulary',
    'open world'
]

reversed_keywords = [
    '3d',
    'bev',
    'active detection',
    'boundary detection',
    'anomaly',
    'oriented',
    'point cloud',
    'video instance segmentation',
    'semantic segmentation',
    'tracking',
    'video object',
    'video',
    'attribute recognition',
    '4d instance segmentation',
    'salient object detection',
    'pose estimation',
    'lidar',
    'acoustic',
    'few-shot',
    'cross-domain',
    'cross domain',
    'domain adaptive',
    'domain adaptation',
    'adaptation',
    'attacks',
    'graph generation',
    'video segmentation'
]

paper_infos = load_paper_info(src_file)
paper_infos = filter_papers(paper_infos, keywords, reversed_keywords)

print('The total number of papers is', len(paper_infos))
if len(paper_infos) > 0:
    df = pd.DataFrame.from_dict(paper_infos)
    df.to_csv(f'filted_cvpr2023.csv', index=True, header=True)
