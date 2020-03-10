from models.resnet_rnn import ResNetEncoder, RNNDecoder


def get_model(args):
    print(f"Fetching model {args.model_name}")
    encoder = ResNetEncoder(args.model_name,
                            embed_dim=args.embed_dim,
                            drop_p=args.dropout,
                            pretrained=args.pretrained)
    
    decoder = RNNDecoder(embed_dim=args.embed_dim,
                         drop_p=args.dropout,
                         num_classes=args.num_classes)
    return (encoder, decoder)