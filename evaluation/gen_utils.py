import os, sys
import numpy as np
from scipy.spatial.distance import cdist
import scipy.io as sio


def euclidean_dist(X_probe, X_gallery):
    return cdist(X_probe, X_gallery)


def get_cam_settings(settings):
    if settings["mode"] == "indoor":
        gallery_cams, probe_cams = [1, 2], [3, 6]
    elif settings["mode"] == "all":
        gallery_cams, probe_cams = [1, 2, 4, 5], [3, 6]
    else:
        assert False, "unknown search mode : " + settings["mode"]

    return gallery_cams, probe_cams


def get_test_ids():
    test_id_filepath = "./data_split/test_id.mat"
    filecontents = sio.loadmat(test_id_filepath)
    test_ids = filecontents["id"].squeeze()

    # make the test ids 0-based
    test_ids = test_ids - 1
    return test_ids


def get_cam_permutation_indices(all_cams):
    rand_cam_perm_filepath = "./data_split/rand_perm_cam.mat"
    filecontents = sio.loadmat(rand_cam_perm_filepath)
    mat_cam_permutations = filecontents["rand_perm_cam"]

    # buffer to hold all permutations
    all_permutations = {}

    for cam_index in all_cams:
        cam_permutations = mat_cam_permutations[cam_index - 1][0].squeeze()
        cam_name = "cam" + str(cam_index)
        if cam_name not in all_permutations:
            all_permutations[cam_name] = {}

        # logistics
        print("{} ids found in cam {}".format(len(cam_permutations), cam_index))

        # collect permutations for all person indices
        for person_index, rand_permutations in enumerate(cam_permutations):
            all_permutations[cam_name][person_index] = rand_permutations - 1

    return all_permutations


def load_feature_file(cam_feature_file):
    filecontents = sio.loadmat(cam_feature_file)
    mat_features = filecontents["feature_test"].squeeze()
    all_features = {}

    # collect features for each person (total 333 persons)
    for person_index, current_features in enumerate(mat_features):
        all_features[person_index] = current_features

    return all_features


def get_testing_set(
    features,
    cam_permutations,
    test_ids,
    run_index,
    cam_id_locations,
    gallery_cams,
    probe_cams,
    settings,
):
    # cam is indexed from 1 - 6
    # person indices are indexed using 0-based numbers
    X_gallery_rgb, Y_gallery_rgb, cam_gallery_rgb, X_probe_IR, Y_probe_IR, cam_probe_IR = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    number_shot = settings["number_shot"]

    # collect rgb images as gallery
    for cam_index in gallery_cams:
        cam_name = "cam" + str(cam_index)
        current_cam_features = features[cam_name]

        # for all the test ids, collect features
        for test_id in test_ids:
            current_id_features = current_cam_features[test_id]

            if np.any(np.array(current_id_features.shape) == 0):
                continue
            # assert (not np.any(np.array(current_id_features.shape) == 0)), 'test id feature count is 0'

            # get the current permutation
            current_permutation = cam_permutations[cam_name][test_id][run_index]
            current_permutation = current_permutation[:number_shot]

            selected_features = current_id_features[current_permutation, :]

            if len(X_gallery_rgb) == 0:
                X_gallery_rgb = selected_features
            else:
                X_gallery_rgb = np.concatenate(
                    (X_gallery_rgb, selected_features), axis=0
                )

            Y_gallery_rgb += [test_id] * number_shot
            cam_gallery_rgb += [cam_id_locations[cam_index - 1]] * number_shot

    Y_gallery_rgb = np.array(Y_gallery_rgb, dtype=np.int)
    cam_gallery_rgb = np.array(cam_gallery_rgb, dtype=np.int)

    # collect all the IR
    for cam_index in probe_cams:
        cam_name = "cam" + str(cam_index)
        current_cam_features = features[cam_name]

        # for all the test ids, collect features
        for test_id in test_ids:
            current_id_features = current_cam_features[test_id]
            if np.any(np.array(current_id_features.shape) == 0):
                continue
            # assert len(current_id_features) != 0, 'test id feature count is 0'

            if len(X_probe_IR) == 0:
                X_probe_IR = current_id_features

            else:
                X_probe_IR = np.concatenate((X_probe_IR, current_id_features), axis=0)

            Y_probe_IR += [test_id] * len(current_id_features)
            cam_probe_IR += [cam_id_locations[cam_index - 1]] * len(current_id_features)

    Y_probe_IR = np.array(Y_probe_IR, dtype=np.int)
    cam_probe_IR = np.array(cam_probe_IR, dtype=np.int)

    return (
        X_gallery_rgb,
        Y_gallery_rgb,
        cam_gallery_rgb,
        X_probe_IR,
        Y_probe_IR,
        cam_probe_IR,
    )


def get_unique(array):
    _, idx = np.unique(array, return_index=True)
    return array[np.sort(idx)]


def get_cmc_multi_cam(Y_gallery, cam_gallery, Y_probe, cam_probe, dist):
    # dist = #probe x #gallery
    num_probes, num_gallery = dist.shape
    gallery_unique_count = get_unique(Y_gallery).shape[0]
    match_counter = np.zeros((gallery_unique_count))

    # sort the distance matrix
    sorted_indices = np.argsort(dist, axis=-1)

    Y_result = Y_gallery[sorted_indices]
    cam_locations_result = cam_gallery[sorted_indices]

    valid_probe_sample_count = 0

    for probe_index in range(num_probes):
        # remove gallery samples from the same camera of the probe
        Y_result_i = Y_result[probe_index, :]
        Y_result_i[cam_locations_result[probe_index, :] == cam_probe[probe_index]] = -1

        # remove the -1 entries from the label result
        # print(Y_result_i.shape)
        Y_result_i = np.array([i for i in Y_result_i if i != -1])
        # print(Y_result_i.shape)

        # remove duplicated id in "stable" manner
        Y_result_i_unique = get_unique(Y_result_i)
        # print(Y_result_i_unique, gallery_unique_count)

        # match for probe i
        match_i = Y_result_i_unique == Y_probe[probe_index]

        if np.sum(match_i) != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            match_counter += match_i

    rankk = match_counter / valid_probe_sample_count
    cmc = np.cumsum(rankk)
    return cmc


def get_mAP_multi_cam(Y_gallery, cam_gallery, Y_probe, cam_probe, dist):
    # dist = #probe x #gallery
    num_probes, num_gallery = dist.shape

    # sort the distance matrix
    sorted_indices = np.argsort(dist, axis=-1)

    Y_result = Y_gallery[sorted_indices]
    cam_locations_result = cam_gallery[sorted_indices]

    valid_probe_sample_count = 0
    avg_precision_sum = 0

    for probe_index in range(num_probes):
        # remove gallery samples from the same camera of the probe
        Y_result_i = Y_result[probe_index, :]
        Y_result_i[cam_locations_result[probe_index, :] == cam_probe[probe_index]] = -1

        # remove the -1 entries from the label result
        # print(Y_result_i.shape)
        Y_result_i = np.array([i for i in Y_result_i if i != -1])
        # print(Y_result_i.shape)

        # match for probe i
        match_i = Y_result_i == Y_probe[probe_index]
        true_match_count = np.sum(match_i)

        if true_match_count != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            true_match_rank = np.where(match_i)[0]

            ap = np.mean(
                np.array(range(1, true_match_count + 1)) / (true_match_rank + 1)
            )
            avg_precision_sum += ap

    mAP = avg_precision_sum / valid_probe_sample_count
    return mAP