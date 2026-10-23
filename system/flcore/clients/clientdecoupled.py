import copy
import torch
import torch.nn as nn

from flcore.clients.clientbase import Client


class DecoupledClient(Client):
    """
    Lớp cơ sở Client cho kiến trúc tách rời (Decoupled Backbone/Encoder & Classifier Head):
    - Kế thừa từ Client (clientbase).
    - Quản lý việc bóc tách Encoder và Head tự động.
    - Khởi tạo và đồng bộ 2 optimizer độc lập (opt_encoder, opt_head) với learning rate tương ứng.
    - Khởi tạo 2 learning rate scheduler độc lập nếu bật learning_rate_decay.
    - Cung cấp phương thức dùng chung extract_features(x).
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = getattr(self, 'args', args)

        # 1. Tách Backbone (Encoder) và Classification Head
        self.encoder, self.head = self._split_model()

        # 2. Cấu hình 2 Optimizer tách riêng cho Encoder và Head
        opt_cls = type(self.optimizer)
        def_cfg = copy.deepcopy(self.optimizer.defaults)

        head_params = list(self.head.parameters())
        enc_params = list(self.encoder.parameters())

        encoder_lr = getattr(args, 'encoder_lr', None)
        enc_lr = float(encoder_lr) if (encoder_lr is not None and float(encoder_lr) > 0) else self.learning_rate

        head_lr = getattr(args, 'head_lr', None)
        h_lr = float(head_lr) if (head_lr is not None and float(head_lr) > 0) else self.learning_rate

        if len(head_params) > 0 and len(enc_params) > 0:
            enc_cfg = copy.deepcopy(def_cfg)
            head_cfg = copy.deepcopy(def_cfg)
            enc_cfg['lr'] = enc_lr
            head_cfg['lr'] = h_lr

            self.opt_encoder = opt_cls(enc_params, **enc_cfg)
            self.opt_head = opt_cls(head_params, **head_cfg)
            self.optimizer = self.opt_head  # Đồng bộ self.optimizer cho Base Client
            self.opt_base = self.opt_encoder

            opt_enc_params = set(p for group in self.opt_encoder.param_groups for p in group['params'])
            opt_head_params = set(p for group in self.opt_head.param_groups for p in group['params'])
            assert opt_enc_params.isdisjoint(opt_head_params), "Tham số của 2 optimizer không được trùng lặp!"
            assert (opt_enc_params | opt_head_params) == set(self.model.parameters()), "Hợp tham số 2 optimizer phải bằng toàn bộ tham số model!"

            if self.learning_rate_decay:
                gamma = getattr(self.args, 'learning_rate_decay_gamma', 0.99)
                self.opt_head_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt_head, gamma=gamma)
                self.learning_rate_scheduler = self.opt_head_scheduler
                self.opt_encoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt_encoder, gamma=gamma)
            else:
                self.opt_encoder_scheduler = None
                self.opt_head_scheduler = None
        elif len(head_params) == 0:
            # Fallback nếu head không có trainable params (nn.Identity)
            enc_cfg = copy.deepcopy(def_cfg)
            enc_cfg['lr'] = enc_lr
            self.opt_encoder = opt_cls(enc_params, **enc_cfg)
            self.opt_head = None
            self.optimizer = self.opt_encoder
            self.opt_base = self.opt_encoder

            opt_enc_params = set(p for group in self.opt_encoder.param_groups for p in group['params'])
            opt_head_params = set()
            assert opt_enc_params.isdisjoint(opt_head_params), "Tham số của 2 optimizer không được trùng lặp!"
            assert (opt_enc_params | opt_head_params) == set(self.model.parameters()), "Hợp tham số 2 optimizer phải bằng toàn bộ tham số model!"

            if self.learning_rate_decay:
                gamma = getattr(self.args, 'learning_rate_decay_gamma', 0.99)
                self.opt_encoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt_encoder, gamma=gamma)
                self.learning_rate_scheduler = self.opt_encoder_scheduler
                self.opt_head_scheduler = None
            else:
                self.opt_encoder_scheduler = None
                self.opt_head_scheduler = None
        else:
            head_cfg = copy.deepcopy(def_cfg)
            head_cfg['lr'] = h_lr
            self.opt_encoder = None
            self.opt_head = opt_cls(head_params, **head_cfg)
            self.optimizer = self.opt_head
            self.opt_base = None

            opt_enc_params = set()
            opt_head_params = set(p for group in self.opt_head.param_groups for p in group['params'])
            assert opt_enc_params.isdisjoint(opt_head_params), "Tham số của 2 optimizer không được trùng lặp!"
            assert (opt_enc_params | opt_head_params) == set(self.model.parameters()), "Hợp tham số 2 optimizer phải bằng toàn bộ tham số model!"

            if self.learning_rate_decay:
                gamma = getattr(self.args, 'learning_rate_decay_gamma', 0.99)
                self.opt_head_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt_head, gamma=gamma)
                self.learning_rate_scheduler = self.opt_head_scheduler
                self.opt_encoder_scheduler = None
            else:
                self.opt_encoder_scheduler = None
                self.opt_head_scheduler = None

    def _split_model(self):
        """
        Tách backbone (encoder trích xuất đặc trưng) và head phân lớp:
        - Ưu tiên 1: model.base / model.head (BaseHeadSplit)
        - Ưu tiên 2: model.features / model.classifier
        - Fallback: encoder = model, head = nn.Identity()
        """
        if hasattr(self.model, 'base') and hasattr(self.model, 'head'):
            encoder, head = self.model.base, self.model.head
        elif hasattr(self.model, 'features') and hasattr(self.model, 'classifier'):
            encoder, head = self.model.features, self.model.classifier
        else:
            encoder, head = self.model, nn.Identity()

        enc_params = set(encoder.parameters())
        head_params = set(head.parameters())
        model_params = set(self.model.parameters())

        assert enc_params.isdisjoint(head_params), "Tham số của encoder và head phải rời nhau!"
        assert (enc_params | head_params) == model_params, "Hợp tham số encoder và head phải bằng toàn bộ tham số model!"

        return encoder, head

    def extract_features(self, x):
        """
        Trích xuất biểu diễn đặc trưng rep và logits out:
        - Xử lý input: x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
        - rep_raw = self.encoder(x)
        - rep = rep_raw.view(rep_raw.size(0), -1)
        - out = self.head(rep_raw) (kèm fallback self.head(rep) nếu runtime lỗi chiều tensor)
        - Trả về tuple (rep, out)
        """
        x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
        rep_raw = self.encoder(x)
        rep = rep_raw.view(rep_raw.size(0), -1)
        try:
            out = self.head(rep_raw)
        except (RuntimeError, ValueError):
            out = self.head(rep)
        return rep, out
