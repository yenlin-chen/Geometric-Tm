import re, torch

df_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ProtTrans_Encoder:

    def __init__(self, device=df_device, float16_embeddings=False):

        from transformers import T5Tokenizer, T5EncoderModel

        self.device = device
        self.float16_embeddings = float16_embeddings

        self.tokenizer = T5Tokenizer.from_pretrained(
            'Rostlab/prot_t5_xl_uniref50',
            do_lower_case=False
        )
        self.model = T5EncoderModel.from_pretrained(
            'Rostlab/prot_t5_xl_uniref50',
        ).to(self.device)

        self.model.float()# if device=='cpu' else self.model.half()
        self.model.eval()

    def encode(self, sequence):

        if not isinstance(sequence, str):
            raise TypeError(f'`sequence` must be a string, not {type(sequence)}')

        sequence = ' '.join(re.sub(r"[UZOB]", "X", sequence.upper()))

        tokenized_inputs = self.tokenizer(
            sequence, return_tensors='pt'#, add_special_tokens=True
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**tokenized_inputs)

        if self.float16_embeddings:
            return output.last_hidden_state[0,:-1].detach().cpu().half()
        else:
            return output.last_hidden_state[0,:-1].detach().cpu()

    def __call__(self, sequence):
        return self.encode(sequence)

if __name__ == '__main__':

    encoder = ProtTrans_Encoder()

    seq = 'PRTEIXN'

    code = encoder.encode(seq)

    print(code)
    print('Requires grad (should be false):', code.requires_grad)
