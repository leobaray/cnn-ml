document.addEventListener('DOMContentLoaded', () => {
    // Abas para Câmera / Arquivos
    const tabLinks = document.querySelectorAll('.tab-link');
    tabLinks.forEach(link => {
        link.addEventListener('click', () => {
            const tabId = link.getAttribute('data-tab');
            document.querySelectorAll('.tab-content.active').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab-link.active').forEach(l => l.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            link.classList.add('active');
        });
    });

    // Elementos da DOM
    const camBtn = document.getElementById('startCam');
    const snapBtn = document.getElementById('snap');
    const video = document.getElementById('cam');
    const canvas = document.getElementById('canvas');
    const queueEl = document.getElementById('queue');
    const queueStats = document.getElementById('queueStats');
    const queuePlaceholder = document.getElementById('queue-placeholder');
    const sendAll = document.getElementById('sendAll');
    const clearQueue = document.getElementById('clearQueue');
    const converterSelect = document.getElementById('converter_select_batch');
    const hiddenConverter = document.getElementById('hiddenConverter');
    const fileMulti = document.getElementById('fileMulti');
    const addFiles = document.getElementById('addFiles');
    const batchForm = document.getElementById('batchForm');

    if (!batchForm) return; // Só executa o resto se estiver na página de upload

    let mediaStream = null;
    let queue = []; // {name, file, url}

    const showNotification = (message, type = 'info') => {
        const flashContainer = document.querySelector('.flash') || document.createElement('div');
        if (!document.querySelector('.flash')) {
            flashContainer.className = 'flash';
            document.querySelector('.container').prepend(flashContainer);
        }
        const flashItem = document.createElement('div');
        flashItem.className = `flash-item flash-${type}`;
        flashItem.textContent = message;
        flashContainer.appendChild(flashItem);
        setTimeout(() => flashItem.remove(), 4000);
    };

    const updateUI = () => {
        queueStats.textContent = `${queue.length} item(s) na fila`;
        sendAll.disabled = queue.length === 0 || !converterSelect.value;
        queuePlaceholder.style.display = queue.length > 0 ? 'none' : 'flex';
        renderQueue();
    };

    const renderQueue = () => {
        queueEl.innerHTML = ''; // Limpa tudo exceto o placeholder
        if (queuePlaceholder) queueEl.appendChild(queuePlaceholder);
        
        queue.forEach((item, idx) => {
            const fig = document.createElement('figure');
            fig.className = 'tile';
            fig.innerHTML = `
                <a href="${item.url}" target="_blank" rel="noopener" class="tile-image-link">
                    <img src="${item.url}" alt="${item.name}" loading="lazy">
                </a>
                <figcaption>
                    <div class="meta">
                        <span class="name" title="${item.name}">${item.name}</span>
                        <span class="size">${Math.max(1, Math.round(item.file.size/1024))} KB</span>
                    </div>
                    <button type="button" class="btn-danger btn-mini remove-btn" data-idx="${idx}">Remover</button>
                </figcaption>
            `;
            queueEl.appendChild(fig);
        });

        // Adiciona listeners para os novos botões de remover
        queueEl.querySelectorAll('.remove-btn').forEach(btn => {
            btn.onclick = (e) => {
                const i = parseInt(e.target.getAttribute('data-idx'));
                URL.revokeObjectURL(queue[i].url); // Libera memória
                queue.splice(i, 1);
                updateUI();
            };
        });
    };

    const startCamera = async () => {
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
            video.srcObject = mediaStream;
            snapBtn.disabled = false;
        } catch (e) {
            showNotification('Não foi possível acessar a câmera. Verifique as permissões.', 'error');
        }
    };

    const dataURLtoFile = (dataurl, filename) => {
        const arr = dataurl.split(','), mime = arr[0].match(/:(.*?);/)[1];
        const bstr = atob(arr[1]);
        let n = bstr.length;
        const u8arr = new Uint8Array(n);
        while (n--) u8arr[n] = bstr.charCodeAt(n);
        return new File([u8arr], filename, { type: mime });
    };

    const snapPhoto = () => {
        const w = video.videoWidth || 1280;
        const h = video.videoHeight || 720;
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, w, h);
        const ts = new Date().toISOString().replace(/[:\-T]/g, '').slice(0, 15);
        const name = `cam_${ts}.jpg`;
        const data = canvas.toDataURL('image/jpeg', 0.92);
        const file = dataURLtoFile(data, name);
        const url = URL.createObjectURL(file);
        queue.push({ name, file, url });
        updateUI();
    };

    const addFilesFromInput = () => {
        if (!fileMulti.files || fileMulti.files.length === 0) {
            showNotification('Selecione um ou mais arquivos de imagem.', 'info');
            return;
        }
        Array.from(fileMulti.files).forEach(f => {
            const url = URL.createObjectURL(f);
            queue.push({ name: f.name, file: f, url });
        });
        fileMulti.value = ''; // Limpa o seletor
        updateUI();
    };

    camBtn.onclick = startCamera;
    snapBtn.onclick = snapPhoto;
    addFiles.onclick = addFilesFromInput;

    converterSelect.onchange = () => {
        hiddenConverter.value = converterSelect.value;
        updateUI();
    };

    clearQueue.onclick = () => {
        queue.forEach(i => URL.revokeObjectURL(i.url));
        queue = [];
        updateUI();
    };

    batchForm.addEventListener('submit', (e) => {
        if (queue.length === 0 || !converterSelect.value) {
            e.preventDefault();
            showNotification('Selecione o conversor e adicione fotos na fila.', 'error');
            return;
        }
        // Hack para anexar arquivos dinamicamente ao form para envio padrão
        const dataTransfer = new DataTransfer();
        queue.forEach(item => dataTransfer.items.add(item.file));
        
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.name = 'photos';
        fileInput.multiple = true;
        fileInput.files = dataTransfer.files;
        fileInput.style.display = 'none';
        
        batchForm.appendChild(fileInput);
    });

    updateUI();
});
