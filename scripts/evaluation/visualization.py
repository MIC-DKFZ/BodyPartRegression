
######################### TODO ##############################
def plot_score2index_xyfix( # TODO 
    self,
    dataset,
    preds,
    zs,
    landmark_anchor=3,
    ids=np.arange(0, 20),
    save_path=False,
):
    landmark_xy = self.get_landmark_xy(dataset, preds, 3)

    plt.figure(figsize=(12, 8))
    for i in ids:
        y = preds[i]
        if i not in landmark_xy.keys():
            continue

        z = zs[i]
        x = [z * i for i in range(len(y))]

        x_shift = landmark_xy[i]["x"] * z
        y_shift = landmark_xy[i]["y"]
        x = np.array(x) - x_shift
        y = np.array(y) - y_shift
        plt.plot(x, y)

    plt.grid(True)
    plt.xlabel("z [mm]", fontsize=14)
    plt.ylabel("slice score", fontsize=14)
    plt.title(
        f"Slice scores for volumes in validation dataset\nAnchor: landmark {landmark_anchor} (0, 0)\n",
        fontsize=16,
    )
    if save_path:
        plt.savefig(
            save_path + f"slice-scores-{landmark_anchor}-anchor-xy-00.png",
            dpi=300,
            transparent=False,
            bbox_inches="tight",
        )
    plt.show()

def plot_score2index_xfix( # TODO 
    self,
    dataset,
    preds,
    zs,
    landmark_anchor=3,
    ids=np.arange(0, 20),
    save_path=False,
    title=None,
    colors=[],
):
    landmark_xy = self.get_landmark_xy(dataset, preds, landmark_anchor)
    plotted_ids = []
    if len(colors) == 0:
        colors = self.colors
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, idx in enumerate(ids):
        y = preds[idx]
        color = colors[i % len(colors)]
        if idx not in landmark_xy.keys():
            continue

        z = zs[idx]
        x = [(z * i) for i in range(len(y))]
        x_shift = landmark_xy[idx]["x"] * z
        x = np.array(x) - x_shift

        # get landmark positions
        x_landmarks, y_landmarks = self.landmark_positions(dataset, idx, x, y)

        plt.plot(x / 10, y, label=f"volume {idx}", color=color)
        plt.plot(x_landmarks/10, y_landmarks, color="black", marker=".", linestyle=" ")
        plotted_ids.append(idx)

    # self.set_scientific_style(ax) # TODO 
    plt.xlabel("z [cm]", fontsize=16)
    plt.ylabel("slice score", fontsize=16)
    plt.title(title, fontsize=16)
    if save_path:
        plt.savefig(
            save_path + f"slice-scores-{landmark_anchor}-anchor-x-0.png",
            dpi=300,
            transparent=False,
            bbox_inches="tight",
        )
    plt.show()

    return plotted_ids

def plot_score2index(
    self,
    dataset,
    preds,
    zs,
    ids=np.arange(0, 20),
    save_path=False,
    colors=[],
    title=None,
):
    landmark_xy = self.get_landmark_xy(dataset, preds, 3)
    if len(colors) == 0:
        colors = self.colors

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, idx in enumerate(ids):
        color = colors[i % len(colors)]
        if idx not in landmark_xy.keys():
            continue
        y = preds[idx]
        z = zs[idx]
        x = np.array([(z * i) for i in range(len(y))])

        # get landmark positions
        x_landmarks, y_landmarks = self.landmark_positions(dataset, idx, x, y)

        plt.plot(x / 10, y, label=f"volume {idx}", color=color)
        plt.plot(x_landmarks/10, y_landmarks, color="black", marker=".", linestyle=" ")

    # self.set_scientific_style(ax, legend_anchor=(0.98, 0.7)) # TODO 
    plt.xlabel("z [cm]", fontsize=16)
    plt.ylabel("slice score", fontsize=16)
    plt.title(title, fontsize=16)
    if save_path:
        plt.savefig(
            save_path + f"slice-scores.png",
            dpi=300,
            transparent=False,
            bbox_inches="tight",
        )
    plt.show()


def plot_scores(self, vol_idx, save_path=False):
    (
        slice_idx,
        landmarks_idx,
        x,
        predicted_scores,
        expected_scores,
        errors,
    ) = self.landmarks2score(vol_idx, self.val_dataset, self.train_lm_summary)
    plt.figure(figsize=(15, 8))
    plt.plot(
        np.array(x),
        np.array(predicted_scores),
        linestyle="-",
        label="predicted slice score",
    )
    expected_f = interpolate.interp1d(slice_idx, expected_scores, kind="linear")
    plt.errorbar(
        slice_idx,
        expected_scores,
        yerr=errors,
        marker="x",
        color="orange",
        linestyle="",
        label="expected slice score (from training set)",
    )
    plt.plot(np.array(x), expected_f(np.array(x)), color="orange", linestyle="--")

    for i, landmark in enumerate(landmarks_idx):
        plt.annotate(
            landmark,
            xy=(
                slice_idx[i] + 0.1,
                expected_scores[i] + np.max(expected_scores) * 0.05,
            ),
        )

    plt.grid(True, axis="y")
    plt.legend(loc=0, fontsize=14)
    plt.title(f"Slice scores for volume {vol_idx}\n", fontsize=14)
    plt.xlabel("slice index", fontsize=14)
    plt.ylabel("slice score", fontsize=14)
    if save_path:
        plt.savefig(
            save_path + f"predicted-vs-estiamted-slice-score-vol{vol_idx}.png",
            dpi=300,
            transparent=False,
            bbox_inches="tight",
        )
    plt.show()
    torch.cuda.empty_cache()

def plot_pred2expected_scores(self, save_path=False, ids=np.arange(0, 65)):
    plt.figure(figsize=(14, 8))
    for i in tqdm(ids):  # len(val_dataset)
        (
            slice_idx,
            landmarks_idx,
            x,
            predicted_scores,
            expected_scores,
            errors,
        ) = self.landmarks2score(i, self.val_dataset, self.train_lm_summary)
        if (len(np.unique(slice_idx)) != len(slice_idx)) or len(slice_idx) < 4:
            continue

        expected_f = interpolate.interp1d(slice_idx, expected_scores, kind="linear")
        label = (
            str(i)
            + "_"
            + self.val_dataset.landmarks[i]["filename"][0:8] # TODO Outdated
            + self.val_dataset.landmarks[i]["filename"][-10:]
            .replace(".npy", "")
            .replace("0", "")
        )
        plt.plot(expected_f(x), predicted_scores, label=label)

    plt.grid(True)

    xrange = np.arange(-7, 7)
    plt.plot(xrange, xrange, linestyle="--")
    plt.legend(loc=0)
    plt.xlabel("estimated slice score", fontsize=14)
    plt.ylabel("predicted slice score", fontsize=14)
    if save_path:
        plt.savefig(
            save_path + f"predicted-vs-estiamted-slice-score-multiple-volumes.png",
            dpi=300,
            transparent=False,
            bbox_inches="tight",
        )
    plt.show()




def get_landmark_xy(self, dataset, preds, landmark): # TODO 
    landmark_xy = {}
    for i, myDict in dataset.landmarks.items():
        index = myDict["dataset_index"]
        if not index in preds.keys():
            continue

        ys = preds[index]
        if not landmark in myDict["defined_landmarks_i"]:
            continue
        x = myDict["slice_indices"][
            np.where(myDict["defined_landmarks_i"] == landmark)[0][0]
        ]
        y = ys[x]

        landmark_xy[index] = {}
        landmark_xy[index]["x"] = x
        landmark_xy[index]["y"] = y
    return landmark_xy



def landmark_positions(self, dataset, dataset_idx, x, y):
    landmarks = dataset.landmarks 
    landmark_dict_idx = [i for i in range(len(dataset)) if landmarks[i]["dataset_index"] == dataset_idx][0]
    indices = landmarks[landmark_dict_idx]["slice_indices"]
    x_landmarks = x[indices]
    y_landmarks = np.array(y)[indices]

    return x_landmarks, y_landmarks