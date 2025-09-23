import math

# -------------------------
# Fitness Function: Otsu's between-class variance
# -------------------------
def otsu_fitness(image, thresholds):
    thresholds = np.sort(np.array(thresholds, dtype=np.uint8))
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    total = image.size

    thresholds = np.concatenate(([0], thresholds, [255]))
    between_var = 0

    for i in range(len(thresholds) - 1):
        lower, upper = thresholds[i], thresholds[i + 1]
        prob = np.sum(hist[lower:upper + 1]) / total
        if prob == 0:
            continue
        mean = np.sum(np.arange(lower, upper + 1) * hist[lower:upper + 1]) / (np.sum(hist[lower:upper + 1]) + 1e-6)
        between_var += prob * mean**2

    return between_var

# -------------------------
# Lévy flight
# -------------------------
def levy_flight(Lambda=1.5):
    sigma = ( (math.gamma(1+Lambda) * np.sin(math.pi*Lambda/2)) /
              (math.gamma((1+Lambda)/2) * Lambda * 2**((Lambda-1)/2)) ) ** (1/Lambda)
    u = np.random.normal(0, sigma, 1)
    v = np.random.normal(0, 1, 1)
    step = u / (abs(v)**(1/Lambda))
    return step[0]

# -------------------------
# Cuckoo Search Algorithm
# -------------------------
def cuckoo_search(image, n=15, max_gen=10, pa=0.25, num_thresh=2):
    nests = np.random.randint(0, 256, size=(n, num_thresh))
    fitness = np.array([otsu_fitness(image, t) for t in nests])

    best_nest = nests[np.argmax(fitness)]
    best_fitness = np.max(fitness)

    for gen in range(max_gen):
        for i in range(n):
            step_size = levy_flight()
            new_solution = nests[i] + step_size * (nests[i] - best_nest)
            new_solution = np.clip(new_solution, 0, 255).astype(int)

            f_new = otsu_fitness(image, new_solution)
            if f_new > fitness[i]:
                nests[i] = new_solution
                fitness[i] = f_new

        # Abandon fraction of nests
        num_abandon = int(pa * n)
        abandon_idx = np.random.choice(n, num_abandon, replace=False)
        nests[abandon_idx] = np.random.randint(0, 256, size=(num_abandon, num_thresh))
        fitness[abandon_idx] = [otsu_fitness(image, t) for t in nests[abandon_idx]]

        best_nest = nests[np.argmax(fitness)]
        best_fitness = np.max(fitness)
        print(f"Generation {gen+1}: Best Thresholds = {best_nest}, Fitness = {best_fitness:.6f}")

    return best_nest

# -------------------------
# Segment Image Using Thresholds
# -------------------------
def segment_image(image, thresholds):
    thresholds = np.sort(np.array(thresholds))
    segmented = np.zeros_like(image)
    prev = 0
    for t in thresholds:
        segmented[(image >= prev) & (image < t)] = t
        prev = t
    segmented[image >= thresholds[-1]] = 255
    return segmented

# -------------------------
# Run Segmentation on CSV Images
# -------------------------
df = pd.read_csv('/content/train.csv')

for idx, row in df.iterrows():
    image_filename = row['images']
    image_path = f'/content/images/{image_filename}'

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not read {image_path}")
        continue

    print(f"\nSegmenting Image: {image_filename}, shape: {image.shape}")

    num_thresholds = 3
    best_thresholds = cuckoo_search(image, n=20, max_gen=10, num_thresh=num_thresholds)
    segmented = segment_image(image, best_thresholds)

    # Display
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(image, cmap='gray')
    plt.subplot(1,2,2)
    plt.title(f"Segmented (Thresh={best_thresholds})")
    plt.imshow(segmented, cmap='gray')
    plt.show()
