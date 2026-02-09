# paged_llama/llama/configuration_utils.py

class PreTrainedConfig:
    """
    모든 설정(Config) 클래스의 부모가 되는 기본 클래스입니다.
    """
    def __init__(self, **kwargs):
        # 기본 속성들 초기화
        self.return_dict = kwargs.pop("return_dict", True)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.torchscript = kwargs.pop("torchscript", False)
        self.use_bfloat16 = kwargs.pop("use_bfloat16", False)
        self.pruned_heads = kwargs.pop("pruned_heads", {})
        self.tie_word_embeddings = kwargs.pop("tie_word_embeddings", True)
        
        # 나머지 인자들을 속성으로 설정
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError:
                pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        임시로 구현한 from_pretrained 메서드입니다.
        실제로는 파일을 읽어야 하지만, 지금은 기본 설정으로 객체를 반환합니다.
        """
        return cls(**kwargs)

    def to_dict(self):
        """
        설정을 딕셔너리로 변환하는 메서드
        """
        output = copy.deepcopy(self.__dict__)
        return output