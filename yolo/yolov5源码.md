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