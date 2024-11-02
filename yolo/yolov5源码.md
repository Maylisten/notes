# Yolov5 源码
## 数据集
### LoadImagesAndLabels(Dataset)
加载图片以及对应的 label
```python
class LoadImagesAndLabels(Dataset):
	# 缓存的版本号，更改版本号可以强制重新生成缓存
    cache_version = 0.6
    
    # 在数据增强或图像缩放时，可以从这个列表中随机选取一种插值方法。这些方法在缩放时对图像质量有不同影响，可以增加数据的多样性，提高模型的泛化能力。
	# cv2.INTER_NEAREST：最近邻插值，速度快但效果不平滑。
	# cv2.INTER_LINEAR：双线性插值，适用于缩小图像，较平滑。
	# cv2.INTER_CUBIC：双三次插值，计算复杂度高，效果更平滑，适用于放大。
	# cv2.INTER_AREA：区域插值，适用于缩小图像，能较好地减少混叠。
	# cv2.INTER_LANCZOS4：Lanczos插值，适合对图像放大，质量较高，但计算较慢。
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]  
  
    def __init__(  
        self,  
        path,  
        img_size=640,  
        batch_size=16,  
        augment=False,  
        hyp=None,  
        rect=False,  
        image_weights=False,  
        cache_images=False,   # 是否缓存图像到内存
        single_cls=False,  # 是否将所有标签视为单一类别
        stride=32,  
        pad=0.0,  # 填充率，用于图像的填充操作。当图像尺寸不符合期望的形状时，使用该参数指定的比例来填充边缘
        min_items=0,  # 最小标签数，用于过滤图像，只加载标签数量大于等于该值的图像
        prefix="",  # 日志信息的前缀，
        rank=-1,  # 当前进程的分布式训练编号
        seed=0,  # 随机种子，用于控制随机操作（如数据增强、采样等）的一致性，使得结果可重复。
    ):  
	    # 目标图像尺寸，用于将输入图像统一缩放到此大小
        self.img_size = img_size  
        # 是否需要做数据增强
        self.augment = augment  
        # 超参数
        self.hyp = hyp  
        # 图片权重
        self.image_weights = image_weights  
        # 是否进行矩形训练（Rectangular Training），矩形训练能将图像按长宽比进行排序，以减少填充操作。
        self.rect = False if image_weights else rect  
        # 控制是否进行马赛克增强，将四张图片组合成一张
        self.mosaic = self.augment and not self.rect   
        # 指定马赛克拼接图像的边界
        self.mosaic_border = [-img_size // 2, -img_size // 2]  
        # stride 表示模型下采样的步幅
        self.stride = stride  
        # 数据集路径，即存储图像和标签的目录
        self.path = path  
        # 初始化 Albumentations 数据增强库
        # Albumentations 是一个流行的数据增强库， 支持各种图像增强方法，如旋转、裁剪、亮度调整等，可以丰富训练数据，提高模型的泛化能力
        self.albumentations = Albumentations(size=img_size) if augment else None  

		# 读取所有的图片路径（存储到 f 数组中）
        try:  
            f = []
            for p in path if isinstance(path, list) else [path]:  
                p = Path(p)   
                if p.is_dir():
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)  
                elif p.is_file():
                    with open(p) as t:  
                        t = t.read().strip().splitlines()  
                        parent = str(p.parent) + os.sep  
                        f += [x.replace("./", parent, 1) if x.startswith("./") else x for x in t]  # to global path  
                    raise FileNotFoundError(f"{prefix}{p} does not exist")  
            self.im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)  
            assert self.im_files, f"{prefix}No images found"  
        except Exception as e:  
            raise Exception(f"{prefix}Error loading data from {path}: {e}\n{HELP_URL}") from e  
  
        # 获取图片对应的 label 数组
        self.label_files = img2label_paths(self.im_files)
        # 得到 cache 文件的路径  
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix(".cache")  
        # 读取cache 的内容
        try:  
	        # cache 存在，读取缓存
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True 
            assert cache["version"] == self.cache_version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
        except Exception: 
	        # cache 不存在，生成缓存 
            cache, exists = self.cache_labels(cache_path, prefix), False  
  
        # 检查缓存文件的有效性      
        nf, nm, ne, nc, n = cache.pop("results")
        if exists and LOCAL_RANK in {-1, 0}:  
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"            
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)
            if cache["msgs"]:  
                LOGGER.info("\n".join(cache["msgs"]))  
        assert nf > 0 or not augment, f"{prefix}No labels found in {cache_path}, can not start training. {HELP_URL}"  
  
        # 删除缓存中的 hash, version, msgs 信息
        [cache.pop(k) for k in ("hash", "version", "msgs")] 
        # 解构赋值，得到 labels（每张图片的标签）和 shapes（每张图片的 shape）
        labels, shapes, self.segments = zip(*cache.values())  
        # 所有标签的数量
        nl = len(np.concatenate(labels, 0))
        # 没有标签警告
        assert nl > 0 or not augment, f"{prefix}All labels empty in {cache_path}, can not start training. {HELP_URL}"  
        # 每张图片的标签
        self.labels = list(labels)  
        # 每张图片的 shape
        self.shapes = np.array(shapes)  
        # 每张图片的路径
        self.im_files = list(cache.keys())
        # 每张图片的标签文件路径
        self.label_files = img2label_paths(cache.keys())
  
        # 过滤掉图片标签小于 min_items 的图片       
        if min_items:  
            include = np.array([len(x) >= min_items for x in self.labels]).nonzero()[0].astype(int)  
            LOGGER.info(f"{prefix}{n - len(include)}/{n} images filtered from dataset")  
            self.im_files = [self.im_files[i] for i in include]  
            self.label_files = [self.label_files[i] for i in include]  
            self.labels = [self.labels[i] for i in include]  
            self.segments = [self.segments[i] for i in include]  
            self.shapes = self.shapes[include]
		
        # 图片数量      
        n = len(self.shapes)
        # 每张图片对应的 batch 索引
        bi = np.floor(np.arange(n) / batch_size).astype(int) 
        # batch 数量
        nb = bi[-1] + 1
        # 每张图片对应的 batch 索引
        self.batch = bi
        # 图片的总数
        self.n = n  
        # 图片的索引数组
        self.indices = np.arange(n)  
        # 分布式
        if rank > -1: 
            self.indices = self.indices[np.random.RandomState(seed=seed).permutation(n) % WORLD_SIZE == RANK]  
  
        # 用于将类别不在 include_class 中的标签删除，默认 include_class 为空
        include_class = []
        self.segments = list(self.segments)  
        include_class_array = np.array(include_class).reshape(1, -1)  
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):  
            if include_class:  
                j = (label[:, 0:1] == include_class_array).any(1)  
                self.labels[i] = label[j]  
                if segment:  
                    self.segments[i] = [segment[idx] for idx, elem in enumerate(j) if elem]  
            if single_cls: 
	             # 将所有类别标签设置为 0，实现单类别训练
                self.labels[i][:, 0] = 0  
  
        # 矩形训练（Rectangular Training），对所有图像按宽高比进行计算和排序，然后将宽高比相近的图像放入同一个批次
        if self.rect:  
            s = self.shapes  # wh  
            ar = s[:, 1] / s[:, 0]  
            irect = ar.argsort()  
            self.im_files = [self.im_files[i] for i in irect]  
            self.label_files = [self.label_files[i] for i in irect]  
            self.labels = [self.labels[i] for i in irect]  
            self.segments = [self.segments[i] for i in irect]  
            self.shapes = s[irect]
            ar = ar[irect]  
            shapes = [[1, 1]] * nb  
            for i in range(nb):  
                ari = ar[bi == i]  
                mini, maxi = ari.min(), ari.max()  
                if maxi < 1:  
                    shapes[i] = [maxi, 1]  
                elif mini > 1:  
                    shapes[i] = [1, 1 / mini]  
  
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride  
  
        # 将 images 缓存到 内存/磁盘
        if cache_images == "ram" and not self.check_cache_ram(prefix=prefix):  
            cache_images = False  
        self.ims = [None] * n  
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]  
        if cache_images:  
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes  
            self.im_hw0, self.im_hw = [None] * n, [None] * n  
            fcn = self.cache_images_to_disk if cache_images == "disk" else self.load_image  
            results = ThreadPool(NUM_THREADS).imap(lambda i: (i, fcn(i)), self.indices)  
            pbar = tqdm(results, total=len(self.indices), bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)  
            for i, x in pbar:  
                if cache_images == "disk":  
                    b += self.npy_files[i].stat().st_size  
                else:  # 'ram'  
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)  
                    b += self.ims[i].nbytes * WORLD_SIZE  
                pbar.desc = f"{prefix}Caching images ({b / gb:.1f}GB {cache_images})"  
            pbar.close()  
  
    def __len__(self):  
        """Returns the number of images in the dataset."""  
        return len(self.im_files)  

    def __getitem__(self, index):  

		# 图片的索引
        index = self.indices[index]  # linear, shuffled, or image_weights  
		# 超参数
        hyp = self.hyp  
        # 是否需要 mosaic
        mosaic = self.mosaic and random.random() < hyp["mosaic"]  
        if mosaic:  
            # img：由当前和随机三张共四张图片拼接的成的马赛克图，以及转换合并后的 xyxy 格式labels
            img, labels = self.load_mosaic(index)  
            shapes = None  
  
            if random.random() < hyp["mixup"]:  
	            # mixup 数据增强，将当前图像 img 和其标签 labels 与另一张随机选取的马赛克拼接图进行混合，生成新的增强图像和标签
                img, labels = mixup(img, labels, *self.load_mosaic(random.choice(self.indices)))  
  
        else:  
            # 加载图片  
            img, (h0, w0), (h, w) = self.load_image(index)  
  
            # Letterbox  
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape  
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)  
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling  
			
            labels = self.labels[index].copy()  
            if labels.size:
	            # 将相对于图片宽高比例的 xywh 转为相对于图片左上角像素的 xyxy  
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])  
  
            if self.augment:  
	            # random_perspective 数据增强
                img, labels = random_perspective(  
                    img,  
                    labels,  
                    degrees=hyp["degrees"],  
                    translate=hyp["translate"],  
                    scale=hyp["scale"],  
                    shear=hyp["shear"],  
                    perspective=hyp["perspective"],  
                )  

		# labels的数量  
        nl = len(labels)  
        
        if nl: 
		    # 归一化处理，转为 xywh 的格式
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)  

		# 一系列的数据增强操作
        if self.augment:  
            # Albumentations  
            img, labels = self.albumentations(img, labels)  
            nl = len(labels)  # update after albumentations  
  
            # HSV color-space            
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])  
  
            # Flip up-down  
            if random.random() < hyp["flipud"]:  
                img = np.flipud(img)  
                if nl:  
                    labels[:, 2] = 1 - labels[:, 2]  
  
            # Flip left-right  
            if random.random() < hyp["fliplr"]:  
                img = np.fliplr(img)  
                if nl:  
                    labels[:, 1] = 1 - labels[:, 1]  
  
            # Cutouts  
            # labels = cutout(img, labels, p=0.5)            
            # nl = len(labels)  # update after cutout  

		
		labels_out = torch.zeros((nl, 6))  
		if nl:  
			labels_out[:, 1:] = torch.from_numpy(labels)  

		# 调整图像通道
		img = img.transpose((2, 0, 1))[::-1] 
		img = np.ascontiguousarray(img)  

		# 返回数据增强后的图像(马赛克图)、归一化的 labels (n,1+xywh)，图片的路径，图片的shape
        return torch.from_numpy(img), labels_out, self.im_files[index], shapes  
	
	# 接收一个索引 i，返回（调整后的图像、图像的原始高度宽度 (hw)、以及调整后的高度宽度）
	# 调整后的图像最大边和目标尺寸相同
    def load_image(self, i):  
	    # im：图片（BGR）
	    # f：图片文件路径
	    # fn：图像对应的 .npy 文件路径
        im, f, fn = (  
            self.ims[i],  
            self.im_files[i],  
            self.npy_files[i],  
        )  
        # 图片不在 RAM 缓存中 
        if im is None:  
            if fn.exists(): 
	            # 如果 npy 文件存在，则 load npy
                im = np.load(fn)  
            else: 
	            # 从文件中读取， BGR 格式
                im = cv2.imread(f) 
                # 找不到文件则抛出错误
                assert im is not None, f"Image Not Found {f}"  
            # 图片原始的高和宽  
            h0, w0 = im.shape[:2]
            # 计算缩放比例 r
            r = self.img_size / max(h0, w0) 
            # 需要调整尺寸
            if r != 1:  
	            # 使用 (cv2.INTER_LINEAR) 或 cv2.INTER_AREA 方法进行插值
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA  
                # 使用 cv2.resize 函数将图像调整为目标尺寸 (w0 * r, h0 * r)，按比例缩放宽度和高度。
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)  
            return im, (h0, w0), im.shape[:2] 
        # 在缓存中则直接返回缓存的值
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized  
  
    def cache_images_to_disk(self, i):  
        """Saves an image to disk as an *.npy file for quicker loading, identified by index `i`."""  
        f = self.npy_files[i]  
        if not f.exists():  
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))  
  
    def load_mosaic(self, index):  
	    # labels4：存储 4 张拼接图像的标签
        labels4, segments4 = [], []
        # 单张图片的尺寸
        s = self.img_size  
        # 四张拼接图片的中心点坐标
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y  
        # 随机选取 3 另外三张图片
        indices = [index] + random.choices(self.indices, k=3)  
        # 打乱三张图片
        random.shuffle(indices) 
        # 拼接图片
        for i, index in enumerate(indices):  
            # 加载图片，得到调整过尺寸后的图片和图片宽高
            img, _, (h, w) = self.load_image(index)  
            
            if i == 0:  
			    # 初始化马赛克图，默认颜色为灰色 rgb(114, 114, 114)
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8) 
                # top left， 图片是尽可能靠近中心的，即图片过大则裁掉 top 和 left 部分，图片过小则填充 top 和 left 部分
                # 图片左上角和右下角两点的坐标（相对于马赛克图左上角）
                # max 是避免图片超过拼接图片的边界
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc 
                # 图片裁剪后的左上角和右下角坐标（相对于原图的左上角）
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  
            elif i == 1: 
		        # top right  
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc  
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h  
            elif i == 2:
	            # bottom left  
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)  
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)  
            elif i == 3:
	            # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)  
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)  

			# 将图片拼接到马赛克图片中
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

			# 需要填充的宽
            padw = x1a - x1b  
            # 需要填充的高
            padh = y1a - y1b  
  
            # 小图对应的 labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()  
            if labels.size:  
	            # 将图片的 labels 从相对于原图宽高百分比的 xywh 转为 相对于马赛克图左上角的像素 xyxy
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]  
            labels4.append(labels)  
            segments4.extend(segments)  
  
        # 将 label 的框限制在拼接图的范围内
        labels4 = np.concatenate(labels4, 0)  
        for x in (labels4[:, 1:], *segments4):  
            np.clip(x, 0, 2 * s, out=x)  
            
        # 数据增强，从已有的分割区域中随机选择一些对象复制并粘贴到目标图像上
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp["copy_paste"])  
        # 是一种随机仿射变换，包括旋转、缩放、平移、倾斜、透视变换等操作。这些操作可让模型更好地适应目标的不同位置、角度和大小，增加图像数据的多样性
        img4, labels4 = random_perspective(  
            img4,  
            labels4,  
            segments4,  
            degrees=self.hyp["degrees"],  
            translate=self.hyp["translate"],  
            scale=self.hyp["scale"],  
            shear=self.hyp["shear"],  
            perspective=self.hyp["perspective"],  
            border=self.mosaic_border,  
        )
		# 返回拼接后的马赛克图，以及转换为马赛克图坐标系下的像素 xyxy
        return img4, labels4  
  
    def load_mosaic9(self, index):  
        """Loads 1 image + 8 random images into a 9-image mosaic for augmented YOLOv5 training, returning labels and  
        segments.        """        labels9, segments9 = [], []  
        s = self.img_size  
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices  
        random.shuffle(indices)  
        hp, wp = -1, -1  # height, width previous  
        for i, index in enumerate(indices):  
            # Load image  
            img, _, (h, w) = self.load_image(index)  
  
            # place img in img9  
            if i == 0:  # center  
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles  
                h0, w0 = h, w  
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates  
            elif i == 1:  # top  
                c = s, s - h, s + w, s  
            elif i == 2:  # top right  
                c = s + wp, s - h, s + wp + w, s  
            elif i == 3:  # right  
                c = s + w0, s, s + w0 + w, s + h  
            elif i == 4:  # bottom right  
                c = s + w0, s + hp, s + w0 + w, s + hp + h  
            elif i == 5:  # bottom  
                c = s + w0 - w, s + h0, s + w0, s + h0 + h  
            elif i == 6:  # bottom left  
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h  
            elif i == 7:  # left  
                c = s - w, s + h0 - h, s, s + h0  
            elif i == 8:  # top left  
                c = s - w, s + h0 - hp - h, s, s + h0 - hp  
  
            padx, pady = c[:2]  
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords  
  
            # Labels            labels, segments = self.labels[index].copy(), self.segments[index].copy()  
            if labels.size:  
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format  
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]  
            labels9.append(labels)  
            segments9.extend(segments)  
  
            # Image  
            img9[y1:y2, x1:x2] = img[y1 - pady :, x1 - padx :]  # img9[ymin:ymax, xmin:xmax]  
            hp, wp = h, w  # height, width previous  
  
        # Offset        yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y  
        img9 = img9[yc : yc + 2 * s, xc : xc + 2 * s]  
  
        # Concat/clip labels  
        labels9 = np.concatenate(labels9, 0)  
        labels9[:, [1, 3]] -= xc  
        labels9[:, [2, 4]] -= yc  
        c = np.array([xc, yc])  # centers  
        segments9 = [x - c for x in segments9]  
  
        for x in (labels9[:, 1:], *segments9):  
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()  
        # img9, labels9 = replicate(img9, labels9)  # replicate  
        # Augment        img9, labels9, segments9 = copy_paste(img9, labels9, segments9, p=self.hyp["copy_paste"])  
        img9, labels9 = random_perspective(  
            img9,  
            labels9,  
            segments9,  
            degrees=self.hyp["degrees"],  
            translate=self.hyp["translate"],  
            scale=self.hyp["scale"],  
            shear=self.hyp["shear"],  
            perspective=self.hyp["perspective"],  
            border=self.mosaic_border,  
        )  # border to remove  
  
        return img9, labels9  
  
    @staticmethod  
    def collate_fn(batch):  
        """Batches images, labels, paths, and shapes, assigning unique indices to targets in merged label tensor."""  
        im, label, path, shapes = zip(*batch)  # transposed  
        for i, lb in enumerate(label):  
            lb[:, 0] = i  # add target image index for build_targets()  
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes  
  
    @staticmethod  
    def collate_fn4(batch):  
        """Bundles a batch's data by quartering the number of shapes and paths, preparing it for model input."""  
        im, label, path, shapes = zip(*batch)  # transposed  
        n = len(shapes) // 4  
        im4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]  
  
        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])  
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])  
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale  
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW  
            i *= 4  
            if random.random() < 0.5:  
                im1 = F.interpolate(im[i].unsqueeze(0).float(), scale_factor=2.0, mode="bilinear", align_corners=False)[  
                    0  
                ].type(im[i].type())  
                lb = label[i]  
            else:  
                im1 = torch.cat((torch.cat((im[i], im[i + 1]), 1), torch.cat((im[i + 2], im[i + 3]), 1)), 2)  
                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s  
            im4.append(im1)  
            label4.append(lb)  
  
        for i, lb in enumerate(label4):  
            lb[:, 0] = i  # add target image index for build_targets()  
  
        return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4
```

## 模型
###  模型结构
![image.png](https://gitee.com/may1234/md-imgs/raw/master/202411010944837.png)
```yaml
# parameters  
nc: 2  # number of classes  
depth_multiple: 0.33  # model depth multiple  
width_multiple: 0.50  # layer channel multiple  
  
# anchors  
anchors:  
  - [10,13, 16,30, 33,23]  # P3/8  
  - [30,61, 62,45, 59,119]  # P4/16  
  - [116,90, 156,198, 373,326]  # P5/32  
  
# YOLOv5 backbone  
backbone:  
  # [from, number, module, args]  
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2  
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4  
   [-1, 3, BottleneckCSP, [128]],  
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8  
   [-1, 9, BottleneckCSP, [256]],  
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16  
   [-1, 9, BottleneckCSP, [512]],  
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32  
   [-1, 1, SPP, [1024, [5, 9, 13]]],  
   [-1, 3, BottleneckCSP, [1024, False]],  # 9  
  ]  
  
# YOLOv5 head  
head:  
  [[-1, 1, Conv, [512, 1, 1]],  
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],  
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4  
   [-1, 3, BottleneckCSP, [512, False]],  # 13  
  
   [-1, 1, Conv, [256, 1, 1]],  
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],  
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3  
   [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)  
  
   [-1, 1, Conv, [256, 3, 2]],  
   [[-1, 14], 1, Concat, [1]],  # cat head P4  
   [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)  
  
   [-1, 1, Conv, [512, 3, 2]],  
   [[-1, 10], 1, Concat, [1]],  # cat head P5  
   [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)  
  
   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)  
  ]
```

### Model
#### parse_model
```python
# d: 模型配置（dict）
# ch：通道数量（list）
def parse_model(d, ch):
	# achors： 3类9种锚框
	# nc：需要检测的类别数量
	# gd（model depth multiple）：控制模型的深度
	# gw（layer channel multiple）：每层卷积通道数量
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']  
	# 每个尺寸特征图的锚框数量
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors  
    # 特征图每个网格对应的参数数量 anchors * (classes + 5) 
    no = na * (nc + 5)
	# layers：
	# savelist： 
	# c2: 图片的通道数量 
    layers, save, c2 = [], [], ch[-1]  
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']): 
		# f(from): 输入来源（下标，比如 -1 代表上一层）
		# n(number): 重复次数
		# m(module): 模块的类型
		# a(args): 模块的超参
	    # eval strings，拿到模型对象
        m = eval(m) if isinstance(m, str) else m

		# 将读取到的字符串（a）转为对象，比如『False』转为 bool 类型
        for j, a in enumerate(args):  
            try:  
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings  
            except:  
                pass  
                
		# 模型的深度（层数），通过 gd 进行控制
        n = max(round(n * gd), 1) if n > 1 else n

		# 对于不同类型的层，格式化不同的参数
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
		    # c1 为输入通道数
		    # c2 为输出通道数
            c1, c2 = ch[f], args[0]
  
			# 除非 c2 == no (yolo 层)，把通道数改为 8 的倍数
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2  

			# 将参数拼接，加上输入通道数和更改后的输出通道数
	        args = [c1, c2, *args[1:]]

			# 对于BottleneckCSP和C3层还插入了一个额外的参数 n（表示重复次数），并将 n 置为 1
            if m in [BottleneckCSP, C3]:  
                args.insert(2, n)  
                n = 1  
        elif m is nn.BatchNorm2d:  
	        # BatchNorm2d 层参数只有输入通道数
            args = [ch[f]]  
        elif m is Concat:  
	        # 拼接层的输出通道数为各个输入特征图的通道数求和
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])  
        elif m is Detect:  
		    # 如果是 Detect 层，将每个特征层的通道数添加到 args 中。如果 args[1] 是整数，则表明该层具有特定数量的锚点（anchors），并按每层的数量扩展为 [list(range(args[1] * 2))] * len(f)。
            args.append([ch[x + 1] for x in f])  
            if isinstance(args[1], int): 
                args[1] = [list(range(args[1] * 2))] * len(f)  
        else:  
		    # 其他层输出通道数等于输入通道数
            c2 = ch[f]  

		# 根据参数和模型类型构建模型  
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        # 得到模型的名称
        t = str(m)[8:-2].replace('__main__.', '')
        # 计算改层的参数数量
        np = sum([x.numel() for x in m_.parameters()]) 
        # 这行代码为 m_ 添加了一些自定义属性：

		# m_.i：当前层的索引 i
		# m_.f：from 索引 f，通常表示前一层的输出通道
		# m_.type：层类型字符串 t
		# m_.np：参数数量 np
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        # 记录哪些层的输出需要保存
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist  
        # 记录每层的模型
        layers.append(m_)  
        # 记录每层的通道数
        ch.append(c2) 
    # 返回网络模型对象和需要保存输出的层的 index
    return nn.Sequential(*layers), sorted(save)
```

#### Focus
```python
# 先拆后合再卷积，加快速度
class Focus(nn.Module):  
    # ch_in, ch_out, kernel, stride, padding, groups，是否需要激活函数
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()  
        # 卷积层
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)  
  
    def forward(self, x):
	    # 先拆后合再卷积
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
```
#### Conv
```python
class Conv(nn.Module):  
     # ch_in, ch_out, kernel, stride, padding, groups，是否需要激活函数  
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True): 
        super(Conv, self).__init__()  
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  
        self.bn = nn.BatchNorm2d(c2)  
        self.act = nn.Hardswish() if act else nn.Identity()  
  
    def forward(self, x):  
        return self.act(self.bn(self.conv(x)))  
  
    def fuseforward(self, x):  
        return self.act(self.conv(x))
```
#### Bottleneck&CSP
```python
# 先对输入特征图做降维，提取关键特征，最后再扩展回到原始通道数
# 这样在不损失信息的情况下，减少了计算开销
class Bottleneck(nn.Module):  
    # ch_in, ch_out, shortcut, groups, expansion  
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, sel1).__init__()  
        c_ = int(c2 * e)  # hidden channels  
        self.cv1 = Conv(c1, c_, 1, 1)  
        self.cv2 = Conv(c_, c2, 3, 1, g=g)  
        self.add = shortcut and c1 == c2  
  
    def forward(self, x):  
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# 融合 Bottleneck 和 CSP，将通过了瓶颈（精简信息）和普通卷积（全局信息）的结果跨阶段分离再融合
class BottleneckCSP(nn.Module):  
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5): 
        super(BottleneckCSP, self).__init__()  
        c_ = int(c2 * e) 
        self.cv1 = Conv(c1, c_, 1, 1)  
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)  
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)  
        self.cv4 = Conv(2 * c_, c2, 1, 1)  
        self.bn = nn.BatchNorm2d(2 * c_) 
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        
    def forward(self, x):  
		# 通过了瓶颈和残差
	    y1 = self.cv3(self.m(self.cv1(x)))  
	    # 直接卷积
	    y2 = self.cv2(x)  
	    return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))
```
#### SPP
```python
# 利用池化提取特征图信息,不会改变特征图大小
class SPP(nn.Module):  
    def __init__(self, c1, c2, k=(5, 9, 13)):  
        super(SPP, self).__init__()  
        c_ = c1 // 2  
        self.cv1 = Conv(c1, c_, 1, 1)  
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)  
        # 池化设计的作用是不改变特征图的大小,方便后续拼接
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])  
  
    def forward(self, x):  
        x = self.cv1(x)
        # 自己 + 三个池化 = 256 * 4
        # 再通过卷积降维到 c2
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
```

#### Detect
```python
class Detect(nn.Module):  
	# 即 锚框像素 / 特征图高度（3个）
    stride = None  
    # onnx export 
    export = False 
  
    def __init__(self, nc=80, anchors=(), ch=()):
        super(Detect, self).__init__()  
        # 检测类别数量
        self.nc = nc
        # 每个特征网格的每个锚框对应的参数数量
        self.no = nc + 5
        # 特征图的数量（共 3 个特征图，9 个锚框）
        self.nl = len(anchors)  
        # 每个特征图对应的锚框数量（3个）
        self.na = len(anchors[0]) // 2 
        # grid 存储二维特征图网格的索引，用于将预测框相对于网格的坐标转为相对于真实图像的
        self.grid = [torch.zeros(1)] * self.nl
        # 格式化 anchors 并调整为张量，（3，3，2）
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)  
        # 将 anchors 信息注册为模型的缓冲区，不会随模型训练改变，但是可以在模型保存和加载时使用
        self.register_buffer('anchors', a)   
        # 将 reshape 后的 anchors 信息注册为模型的缓冲区，适用于未来广播操作，shape 对应特征图（特征图数量，批次，锚框数量，高度，宽度，锚框的宽高）
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)  
        # self.m 是包含多个卷积层的模块列表，每个卷积层用于处理不同通道数的特征图
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv  
  
    def forward(self, x):  
        # 初始化空列表 z，用于存储每个特征图的输出
        z = []
        # 更新 self.training 状态，确保它在导出模式下也保持为 True
        self.training |= self.export  
        # 遍历特征图
        for i in range(self.nl):  
	        # 对每个特征图过一次对应的卷积，将通道数变为 no*na（85 * 3），x[i] 的 shape 为（batch_size, na*no, ny , nx）
            x[i] = self.m[i](x[i])
            # 解构出批次、特征图高、特征图宽
            bs, _, ny, nx = x[i].shape
            # 张量先 reshape 到 (bs, na, no, ny, nx,) ，再调整为 (bs, na, ny, nx, no)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  

			# 如果是 detect 模式
            if not self.training: 
	            # 重新设置网格索引
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:  
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)  
				
				# 特征图通过 sigmoid 归一化
                y = x[i].sigmoid()  
			    # 预测框 xy 转为真实图像，（* 2. - 0.5）是为了将归一化后的结果范围扩大到[-0.5 * 1.5]，增加灵活性
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]
                # * 2) ** 2 也是为了效果更好做的优化
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                # reshape 为（批次，检测框数量，xywhccn）
                z.append(y.view(bs, -1, self.no))  
		# 训练模式返回特征图输出， 如果是检测模式则多返回一个预测框结果
        return x if self.training else (torch.cat(z, 1), x)
```

#### Model
```python
class Model(nn.Module):  
	# model, input channels, number of classes  
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None): 
        super(Model, self).__init__()  
		# 从 data.yaml 中读取到网络的配置，存到 self.yaml
        if isinstance(cfg, dict):  
            self.yaml = cfg
        else: 
            import yaml
            self.yaml_file = Path(cfg).name  
            with open(cfg) as f:  
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)   
	    # 覆盖 yaml 中的 nc 属性
        if nc and nc != self.yaml['nc']:  
            print('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))  
            self.yaml['nc'] = nc  
		
		# model: 读取配置文件生成的神经网络模型
		# save: 需要保存输出的层索引
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out  
        # 最后一层是检测层，防止潜在错误
        if isinstance(m, Detect):  
	        # 假设的图片尺寸
            s = 128
            # 用虚构的图片走一遍网络，得到特征图（3个）尺寸
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  
            # 锚框（像素）/ 特征图宽度（reshape方便广播）= 锚框（相对于特征图网格）
            m.anchors /= m.stride.view(-1, 1, 1)  
            # 检查锚框的顺序
            check_anchor_order(m)  
            # 将比例存储到 Detect 层中
            self.stride = m.stride  
            # 初始化权重和偏置
            self._initialize_biases()

		# 控制台输出一下信息
        self.info()  
        print('')  
  
    def forward(self, x, augment=False, profile=False):  
	    # 数据增强
        if augment:  
            img_size = x.shape[-2:]
            s = [1, 0.83, 0.67]
            f = [None, 3, None]
            y = []
            for si, fi in zip(s, f):  
                xi = scale_img(x.flip(fi) if fi else x, si)  
                yi = self.forward_once(xi)[0] 
                if fi == 2:  
                    yi[..., 1] = img_size[0] - yi[..., 1]
                elif fi == 3:  
                    yi[..., 0] = img_size[1] - yi[..., 0]  
                y.append(yi)  
            return torch.cat(y, 1), None
        else:  
            return self.forward_once(x, profile)  
	# 向前传播的逻辑
    def forward_once(self, x, profile=False):  
	    # y: 存储每一层的输出，用于网络的跳跃连接
	    # dt 用于存储每一层的推理时间，便于性能分析
        y, dt = [], []
        # 每层的序号
        i = 1  
        # 遍历每一层
        for m in self.model:  
	        # 跨层输入
            if m.f != -1: 
	            # 取出前 n 层的输出
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

			# 性能分析
            if profile:  
                try:  
                    import thop  
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS  
                except:  
                    o = 0  
                t = time_synchronized()  
                for _ in range(10):  
                    _ = m(x)  
                dt.append((time_synchronized() - t) * 100)  
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))  
                
            # 调用对应模型的 forward
            x = m(x)
            # 层数加一
            i+=1  
            # 保存当前层的输出结果
            y.append(x if m.i in self.save else None)  # save output  
  
        if profile:  
            print('%.1fms total' % sum(dt))  
        return x  
```
## 训练
#### train
```python
def train(hyp, opt, device, tb_writer=None):  
	# 打印超参
    logger.info(f'Hyperparameters {hyp}') 
    # 输出文件夹
    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(opt.logdir) / 'evolve' 
	# 训练权重输出的文件夹
    wdir = log_dir / 'weights'
    # 训练结果保存路径
    os.makedirs(wdir, exist_ok=True) 
    # 最新训练结果的路径
    last = wdir / 'last.pt'  
    # 最好训练结果的路径
    best = wdir / 'best.pt'  
    # 训练过程中各种指标
    results_file = str(log_dir / 'results.txt')   
    # 轮数、批次大小、总批次、预训练权重、分布式
    epochs, batch_size, total_batch_size, weights, rank = \  
        opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank  
  
    # 保存超参
    with open(log_dir / 'hyp.yaml', 'w') as f:  
        yaml.dump(hyp, f, sort_keys=False)  
    with open(log_dir / 'opt.yaml', 'w') as f:  
        yaml.dump(vars(opt), f, sort_keys=False)  
  
    # 设置 cuda | cpu
    cuda = device.type != 'cpu'  
    # 随机种子  
    init_seeds(2 + rank) 
    # 读取数据集的位置
    with open(opt.data) as f:  
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    # 所有进程都一起  
    with torch_distributed_zero_first(rank):
	    # 检查数据集  
        check_dataset(data_dict) 
    # 训练数据集位置
    train_path = data_dict['train']  # 数据路径与类别名字  
    # 测试数据集位置
    test_path = data_dict['val']  
    # nc：预测类别的数量
    # name: 类别对应的标签
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])
    # 断言检查类别对应的标签数量是否和需要检测的类别数量相同
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  
    # 检查预训练模型是否是.pt
    pretrained = weights.endswith('.pt')  
    # 有预训练模型的话，会自动下载，最好在github下载好 然后放到对应位置  
    if pretrained:  
	    # 分布式环境下保证顺序下载文件
        with torch_distributed_zero_first(rank):  
	        # 尝试从指定地址下载文件
            attempt_download(weights) 
        # 加载 checkpoint，包含 训练的轮数、模型结构、训练参数等
        ckpt = torch.load(weights, map_location=device) 
        if hyp.get('anchors'): 
	        # 如果配置文件中有锚框 (anchors) 参数，强制将预训练权重的 anchors 参数更新为当前设定的 hyp['anchors']
            ckpt['model'].yaml['anchors'] = round(hyp['anchors'])
        # 构建模型
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)
        # 需要排除的模型参数
        exclude = ['anchor'] if opt.cfg or hyp.get('anchors') else [] 
        # 模型参数转为浮点字典
        state_dict = ckpt['model'].float().state_dict()
        # 提取模型需要使用的公共参数，并排除掉一些键值
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect  
        # 模型加载参数
        model.load_state_dict(state_dict, strict=False) 
        # 打印
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))
    else:  
	    # 没有预训练直接构建模型
        model = Model(opt.cfg, ch=3, nc=nc).to(device)  # create 就是咱们之前讲的创建模型那块  
  
    # 累计多少张图片更新一次模型 
    nbs = 64 
    # 累计多少批次更新一次模型 
    accumulate = max(round(nbs / total_batch_size), 1)
    
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay  

	# 创建三个空列表，用于存储不同类型的参数
	pg0, pg1, pg2 = [], [], []
	# 遍历模型中的所有参数
    for k, v in model.named_parameters():  
		# 设置参数为可训练
        v.requires_grad = True  
        # 如果参数名包含'.bias'，表示偏置项，将偏置项添加到pg2列表中
        if '.bias' in k:  
            pg2.append(v)
        elif '.weight' in k and '.bn' not in k:  
		    # 如果是卷积或全连接层的权重（不包含批归一化的权重），将此类权重添加到pg1列表
            pg1.append(v)
        else:  
	        # 其他参数（包括批归一化的权重）添加到pg0列表
            pg0.append(v)

	# 根据配置，创建 adam 或 sgd 学习率优化器
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  
    else:  
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)  

	# 添加 pg1，施加权重衰减
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})
    # 添加 pg2，不施加权重衰减
    optimizer.add_param_group({'params': pg2})
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))  
    del pg0, pg1, pg2  
  
    # 通过余弦退火的方法使学习率在训练的不同阶段逐渐下降
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  
    # plot_lr_scheduler(optimizer, scheduler, epochs)  
  
    # best_fitness是sum([0.0, 0.0, 0.1, 0.9]*[精确度, 召回率, mAP@0.5, mAP@0.5:0.95])  
    # 相当于一个综合指标来判断每一次的得分  
    start_epoch, best_fitness = 0, 0.0  

	# 加载预训练模型，恢复训练状态
    if pretrained:  
        if ckpt['optimizer'] is not None:  
            # 设置优化器
            optimizer.load_state_dict(ckpt['optimizer']) 
	        # checkpoint 中的最佳 fitness 值 
            best_fitness = ckpt['best_fitness']  
            
        if ckpt.get('training_results') is not None:  
            with open(results_file, 'w') as file:  
                file.write(ckpt['training_results'])  # write results.txt  
  
        start_epoch = ckpt['epoch'] + 1  
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)  
            shutil.copytree(wdir, wdir.parent / f'weights_backup_epoch{start_epoch - 1}')
        if epochs < start_epoch:
	        # 若设置的 epoch 小于模型已经训练的轮数，就再训练 epoch 轮
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %  
                        (weights, ckpt['epoch'], epochs))  
            epochs += ckpt['epoch']
  
        del ckpt, state_dict  
  
    # 总的下采样比例（ grid size ）
    gs = int(max(model.stride))
    # 看数据的大小能不能整除这个比例   
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]
  
    # DP mode 如果你的机器里面有过个GPU，需要改一些参数  
    if cuda and rank == -1 and torch.cuda.device_count() > 1:  
        model = torch.nn.DataParallel(model)  
  
    # SyncBatchNorm 多卡同步做BN
    if opt.sync_bn and cuda and rank != -1:  
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)  
        logger.info('Using SyncBatchNorm()')  
  
    # Exponential moving average 滑动平均能让参数更新的更平滑一点不至于波动太大    
    ema = ModelEMA(model) if rank in [-1, 0] else None  
  
    # DDP mode 多机多卡，有时候DP可能会出现负载不均衡，这个能直接解决该问题
    if cuda and rank != -1:  
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)  
  
    # 创建 dataloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,  
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect,  
                                            rank=rank, world_size=opt.world_size, workers=opt.workers)  
    # mlc 为标签中的展现的类别总数                                    
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()
    # 批次总数
    nb = len(dataloader) 
    # 检查 grandtruth 中的 label种类是否小于期望检测的种类
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)  
  
    # 主进程执行 
    if rank in [-1, 0]:  
	    # set EMA updates
        ema.updates = start_epoch * nb // accumulate    
        # 测试数据集
		testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt,  
                                       hyp=hyp, augment=False, cache=opt.cache_images and not opt.notest, rect=True,  
                                       rank=-1, world_size=opt.world_size, workers=opt.workers)[0] 
		# 若是从头开始训练时执行
        if not opt.resume:  
	        # 获取训练集的 labels
            labels = np.concatenate(dataset.labels, 0)  
            # 获取各个 label 的类别
            c = torch.tensor(labels[:, 0]) 
            # 绘制并保存类别分布图
            plot_labels(labels, save_dir=log_dir)  
            # TensorBoard 相关数据可视化
            if tb_writer:  
                tb_writer.add_histogram('classes', c, 0)  
  
            # 自动检查并调整 YOLO 模型的锚框（anchor boxes），使其更适合当前的数据集
            if not opt.noautoanchor:  
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  
  
    # 分类损失的初始权重，hyp['cls'] 的值会被乘以 (nc / 80)，以便在不同类别数的数据集上合理适配分类损失的权重  
    hyp['cls'] *= nc / 80. 
    # 模型中的类别数量
    model.nc = nc
    # 模型的超参数
    model.hyp = hyp
    # 目标损失（obj_loss）与 IoU 损失的比重  
    model.gr = 1.0
    # 根据标签设置各类别数据初始权重  
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)
    # 标签类别的名称
    model.names = names  
  
    # 开始训练
    # 记录时间
    t0 = time.time()  
    # 热身的步数，防止开始学习率过大，至少 1000 步
    nw = max(round(hyp['warmup_epochs'] * nb), 1e3) 
    # 每个类别的 map（平均精度） 
    maps = np.zeros(nc)
    # 精确率, 召回率, mAP@.5, mAP@.5-.95, val_loss(box）,val_loss(obj), val_loss(cls) 
    results = (0, 0, 0, 0, 0, 0, 0)
    # 设置学习率调度器的初始步数
    scheduler.last_epoch = start_epoch - 1   
    # 混合精度训练 
    scaler = amp.GradScaler(enabled=cuda)  
    # 打印信息   
    logger.info('Image sizes %g train, %g test\n'  
                'Using %g dataloader workers\nLogging results to %s\n'  
                'Starting training for %g epochs...' % (imgsz, imgsz_test, dataloader.num_workers, log_dir, epochs))

	# 开始训练
    for epoch in range(start_epoch, epochs):
	    # 模型进入训练状态 
        model.train()  
  
        # 是否要使用图像的权重
        if opt.image_weights:   
            if rank in [-1, 0]:  
	            # 类别的权重，labels 中出现次数越少，权重就高
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2  
                # 各个图片的权重，图片中对应的 labels 类别加权求和得到权重
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)
                # 随机的索引，按照权重有放回的采样，权重高的很容易被多次拿到
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)
            # 分布式环境下同步 indices            、
            if rank != -1:  
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()  
                dist.broadcast(indices, 0)  
                if rank != 0:  
                    dataset.indices = indices.cpu().numpy()  
        
        # 初始化损失
        mloss = torch.zeros(4, device=device)
        # DDP模式每次取数据的随机种子都不同
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)  
        # 创建进度条  
        pbar = enumerate(dataloader)  
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))  
        if rank in [-1, 0]:  
            pbar = tqdm(pbar, total=nb)  # progress bar  
        # 
        optimizer.zero_grad()  
        for i, (imgs, targets, paths, _) in pbar:
	        # imgs: 图像
	        # targets: 标签
	        # paths:图像路径
	        # 记录当前经过的 batch 数量，用于控制 warm-up 期间的学习率和动量。
            ni = i + nb * epoch
            # 图像归一化  
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0  
  
            # Warmup 热身
            # 前 nw 个 batch 中进行 warm-up。  
            if ni <= nw:  
                xi = [0, nw]  # x interp  
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  
                # iou loss ratio (obj_loss = 1.0 or iou)     
                # 控制梯度累积次数
                # np.interp 用线性插值计算 accumulate 的值，以便在 warm-up 阶段逐渐增加
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())  
                # 根据当前的 batch 数量 ni 计算每个参数组的学习率和动量，使其逐渐过渡到预设的初始学习率和动量
                for j, x in enumerate(optimizer.param_groups):    
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])  
                    if 'momentum' in x:  
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])  
  
            # Multi-scale 各种输入的大小
            if opt.multi_scale:  
	            # 图像尺寸随机缩放，范围是[imgsz * 0.5, imgsz * 1.5]
	            # gs 是最终特征图一个网格的大小， //gs * gs 保证图片大小是 gs 的整数倍
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
                # scale 比例  
                sf = sz / max(imgs.shape[2:]) 
                if sf != 1:  # 得到新的输入大小  
	                # 得到图片缩放后的真实大小
	                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  
	                # 以新尺寸 ns 重采样图像
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)  
  
            # 向前传播 
            with amp.autocast(enabled=cuda): 
	            # 预测值，三张不同尺寸的特征图
                pred = model(imgs)
                # 总损失，（分类损失，回归损失，置信度损失）
                loss, loss_items = compute_loss(pred, targets.to(device), model)  
                if rank != -1:  
                    loss *= opt.world_size
  
            # 反向传播，scaler 是混合精度缩放工具
            scaler.scale(loss).backward()  
  
            # 反向传播若干次更新一次参数  
            if ni % accumulate == 0:  
	            # 梯度更新
                scaler.step(optimizer)
                scaler.update()  
                # 清空梯度
                optimizer.zero_grad()  
                # 将当前模型的参数更新到 EMA 模型中
                if ema:  
                    ema.update(model)  
  
            # Print 展示信息  
            if rank in [-1, 0]:  
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses  
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)  
                s = ('%10s' * 2 + '%10.4g' * 6) % (  
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])  
                pbar.set_description(s)  
  
                # Plot  
                if ni < 3:  
                    f = str(log_dir / ('train_batch%g.jpg' % ni))  # filename  
                    result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)  
                    if tb_writer and result is not None:  
                        tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)  
                        # tb_writer.add_graph(model, imgs)  # add model to tensorboard  
  
            # end batch ------------------------------------------------------------------------------------------------  
        # Scheduler 学习率衰减  
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard  
        scheduler.step()  
  
        # DDP process 0 or single-GPU  
        if rank in [-1, 0]:  
            # mAP 更新EMA  
            if ema:  
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])  
            final_epoch = epoch + 1 == epochs  
            if not opt.notest or final_epoch:  # Calculate mAP  
                results, maps, times = test.test(opt.data,  
                                                 batch_size=total_batch_size,  
                                                 imgsz=imgsz_test,  
                                                 model=ema.ema,  
                                                 single_cls=opt.single_cls,  
                                                 dataloader=testloader,  
                                                 save_dir=log_dir,  
                                                 plots=epoch == 0 or final_epoch)  # plot first and last  
  
            # Write            with open(results_file, 'a') as f:  
                f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)  
            if len(opt.name) and opt.bucket:  # 这个整不了，涉及上传  
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))  
  
            # Tensorboard  
            if tb_writer:  
                tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss  
                        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',  
                        'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss  
                        'x/lr0', 'x/lr1', 'x/lr2']  # params  
                for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):  
                    tb_writer.add_scalar(tag, x, epoch)  
  
            # Update best mAP  
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]  
            if fi > best_fitness:  
                best_fitness = fi  
  
            # Save model  
            save = (not opt.nosave) or (final_epoch and not opt.evolve)  
            if save:  
                with open(results_file, 'r') as f:  # create checkpoint  
                    ckpt = {'epoch': epoch,  
                            'best_fitness': best_fitness,  
                            'training_results': f.read(),  
                            'model': ema.ema,  
                            'optimizer': None if final_epoch else optimizer.state_dict()}  
  
                # Save last, best and delete  
                torch.save(ckpt, last)  
                if best_fitness == fi:  
                    torch.save(ckpt, best)  
                del ckpt  
        # end epoch ----------------------------------------------------------------------------------------------------  
    # end training  
    if rank in [-1, 0]:  
        # Strip optimizers  
        n = opt.name if opt.name.isnumeric() else ''  
        fresults, flast, fbest = log_dir / f'results{n}.txt', wdir / f'last{n}.pt', wdir / f'best{n}.pt'  
        for f1, f2 in zip([wdir / 'last.pt', wdir / 'best.pt', results_file], [flast, fbest, fresults]):  
            if os.path.exists(f1):  
                os.rename(f1, f2)  # rename  
                if str(f2).endswith('.pt'):  # is *.pt  
                    strip_optimizer(f2)  # strip optimizer  
                    os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket else None  # upload  
        # Finish        if not opt.evolve:  
            plot_results(save_dir=log_dir)  # save as results.png  
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))  
  
    dist.destroy_process_group() if rank not in [-1, 0] else None  
    torch.cuda.empty_cache()  
    return results
```
