'''
Get list of frames to pull from a video
'''

import skvideo.io

def create_list(vidPath, k, outfile):

    vid = vidPath.split('/')[-1]
    videogen = skvideo.io.vreader(vid)

    with open(outfile, 'w') as f:

        counter = 0
        fnum = 0
        for frame in videogen:
            if fnum % k == 0:
                counter+=1
                f.write('{}_{}\n'.format(vid.split('.')[0], '0'*(7-len(str(fnum))) + str(fnum)))
            fnum += 1

    print('Wrote a total of {} frames'.format(counter))

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--vids-path', default = '', help = 'Path to the directory containing the videos')
    parser.add_argument('--save-path', default = '', help = 'Where to save the frames list')
    parser.add_argument('--k', default = '30', help = 'Saves every "kth" frame')

    args = parser.parse_args()

    create_list(args.vids_path, int(args.k), args.save_path)
