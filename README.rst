=====================================
Running sample deepseek model on 1060
=====================================

I'm curious about DeepSeek and I have GTX 1060. How badly will this go?

Deepseek Code has very nice tutorial on their `github page <https://github.com/deepseek-ai/DeepSeek-Coder?tab=readme-ov-file#1-code-completion>`_

Follow the steps: install the requirements and past the commands in jupyter notebook.

If you want some handholding this video is very helpful -- https://www.youtube.com/watch?v=rlxsDC9aza0

HOWEVER, you will run out of memory.

Use this repo instead if you want to run this very slowly on a low memory system.

 I skip the regular warning about virtualenvs, this worked great with python 3.12.2


Just install the requirements -- :code:`pip install -r requirements.txt`.

:code:`accelerate` is needed to enable memory management so models can be loaded in parts. 

then run the notebook  with

.. code-block:: bash

    jupyter-notebook Untitled.ipynb

This command downloads the models. My average speed on 500Mbit connection was 42MB/s and the models downloaded in about 5 minutes.

.. code-block:: python

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto").cuda()

Then I ran a sample query

.. code-block:: python

    input_text = "#write a quick sort algorithm"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=128)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

On my system I got received quick sort algorithm example in about 5 minutes. lol

.. code-block:: python

    #write a quick sort algorithm

    def quick_sort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[0]
        left = []
        right = []
        for i in range(1, len(arr)):
            if arr[i] < pivot:
                left.append(arr[i])
            else:
                right.append(arr[i])
        return quick_sort(left) + [pivot] + quick_sort(right)


Error log
=========

Initial error
-------------

OutOfMemory error.

.. code-block:: console

    OutOfMemoryError: CUDA out of memory. Tried to allocate 86.00 MiB. GPU 0 has a total capacity of 5.93 GiB of which 34.81 MiB is free. 
    Including non-PyTorch memory, this process has 5.33 GiB memory in use. Of the allocated memory 5.27 GiB is allocated by PyTorch, a
    nd 1.80 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting 
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management 
     (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)


Next error
----------


.. code-block:: console

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto").cuda()


causes this error

..code-block:: console

    ImportError: Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install 'accelerate>=0.26.0'`


I needed to install accelerate package for the device mapper

Next error
----------

I shouldn't run :code:`AutoModelForCausalLM.cuda()` because this loads model straight not memory and I don't have that. So I didnt and everything worked.

.. code-block:: console

    RuntimeError: You can't move a model that has some modules offloaded to cpu or disk.

Just use this command

.. code-block:: python

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")