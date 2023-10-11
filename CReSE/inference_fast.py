import torch



def foward_with_cls_hidden_state_ec(model,
                                    cls_hidden_state_ec,
                                    input_ids_title,
                                    attention_mask_title):
    output_title = model.model_title(input_ids=input_ids_title,
                                    attention_mask=attention_mask_title)
    cls_hidden_state_title = output_title.last_hidden_state[:, 0, :]
    #concatenate hidden state of ecs and titles
    cls_hidden_state_ec_title = torch.cat((cls_hidden_state_ec, cls_hidden_state_title),
                                            dim=1)
    #classify ec-title pair
    logits = model.linear_relu_stack(cls_hidden_state_ec_title)
    
    return logits
