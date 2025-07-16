// 全局变量
let currentVideoInfo = null;
let downloadTasks = {};

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // 绑定事件监听器
    document.getElementById('extractBtn').addEventListener('click', extractVideoInfo);
    document.getElementById('urlInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            extractVideoInfo();
        }
    });
    
    // 初始化工具提示
    initTooltips();
}

function initTooltips() {
    // Bootstrap 工具提示初始化
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// 提取视频信息
async function extractVideoInfo() {
    const url = document.getElementById('urlInput').value.trim();
    
    if (!url) {
        showError('请输入有效的 YouTube 链接');
        return;
    }
    
    if (!isValidYouTubeUrl(url)) {
        showError('请输入有效的 YouTube 链接');
        return;
    }
    
    showLoading();
    hideError();
    
    try {
        const response = await fetch('/extract_info', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            currentVideoInfo = data;
            currentVideoInfo.url = url;
            displayVideoInfo(data);
        } else {
            showError(data.error || '获取视频信息失败');
        }
    } catch (error) {
        showError('网络错误，请检查连接');
        console.error('Extract info error:', error);
    } finally {
        hideLoading();
    }
}

// 验证 YouTube URL
function isValidYouTubeUrl(url) {
    const patterns = [
        /^https?:\/\/(www\.)?youtube\.com\/watch\?v=[\w-]+/,
        /^https?:\/\/(www\.)?youtu\.be\/[\w-]+/,
        /^https?:\/\/(www\.)?m\.youtube\.com\/watch\?v=[\w-]+/
    ];
    return patterns.some(pattern => pattern.test(url));
}

// 显示视频信息
function displayVideoInfo(info) {
    hideWelcome();
    
    // 基本信息
    document.getElementById('videoThumbnail').src = info.thumbnail || '';
    document.getElementById('videoTitle').textContent = info.title || '未知标题';
    document.getElementById('videoUploader').textContent = info.uploader || '未知上传者';
    document.getElementById('videoViews').textContent = formatNumber(info.view_count) || '未知';
    document.getElementById('videoDuration').textContent = formatDuration(info.duration) || '未知';
    document.getElementById('videoDate').textContent = formatDate(info.upload_date) || '未知';
    document.getElementById('videoDescription').textContent = info.description || '无描述';
    
    // 显示格式
    displayFormats(info.formats);
    
    // 显示字幕
    displaySubtitles(info.subtitles);
    
    // 显示视频信息区域
    document.getElementById('videoInfo').style.display = 'block';
    document.getElementById('videoInfo').classList.add('fade-in');
}

// 显示格式选项
function displayFormats(formats) {
    const videoFormats = document.getElementById('videoFormats');
    const audioFormats = document.getElementById('audioFormats');
    
    videoFormats.innerHTML = '';
    audioFormats.innerHTML = '';
    
    // 分类格式
    const videos = formats.filter(f => f.type === 'video');
    const audios = formats.filter(f => f.type === 'audio');
    
    console.log('视频格式数量:', videos.length);
    console.log('音频格式数量:', audios.length);
    
    // 视频格式
    if (videos.length === 0) {
        videoFormats.innerHTML = '<p class="text-muted">无可用视频格式</p>';
    } else {
        videos.forEach(format => {
            const formatElement = createFormatElement(format, 'video');
            videoFormats.appendChild(formatElement);
        });
    }
    
    // 音频格式
    if (audios.length === 0) {
        audioFormats.innerHTML = '<p class="text-muted">无可用音频格式</p>';
    } else {
        audios.forEach(format => {
            const formatElement = createFormatElement(format, 'audio');
            audioFormats.appendChild(formatElement);
        });
    }
    
    // 添加调试按钮
    if (currentVideoInfo && currentVideoInfo.url) {
        const debugButton = document.createElement('button');
        debugButton.className = 'btn btn-outline-info btn-sm mt-2';
        debugButton.innerHTML = '<i class="bi bi-bug"></i> 查看所有格式';
        debugButton.onclick = () => debugFormats(currentVideoInfo.url);
        videoFormats.appendChild(debugButton);
    }
}

// 创建格式元素
function createFormatElement(format, type) {
    const div = document.createElement('div');
    div.className = 'format-item';
    
    let qualityText = format.quality || 'unknown';
    let sizeText = format.filesize_mb > 0 ? `${format.filesize_mb} MB` : '大小未知';
    let extText = format.ext || 'unknown';
    
    let additionalInfo = '';
    let qualityBadge = '';
    
    if (type === 'video') {
        // 视频格式信息
        const height = format.height || 0;
        const resolution = format.resolution || '';
        const fps = format.fps;
        
        // 生成质量标识
        if (height >= 2160) {
            qualityBadge = '<span class="badge bg-danger">4K</span>';
        } else if (height >= 1440) {
            qualityBadge = '<span class="badge bg-warning">2K</span>';
        } else if (height >= 1080) {
            qualityBadge = '<span class="badge bg-success">1080p</span>';
        } else if (height >= 720) {
            qualityBadge = '<span class="badge bg-info">720p</span>';
        } else if (height >= 480) {
            qualityBadge = '<span class="badge bg-secondary">480p</span>';
        } else if (height > 0) {
            qualityBadge = `<span class="badge bg-light text-dark">${height}p</span>`;
        }
        
        additionalInfo = resolution || `${format.width || '?'}x${height || '?'}`;
        if (fps) {
            additionalInfo += ` @ ${fps}fps`;
        }
    } else {
        // 音频格式信息
        const abr = format.abr;
        if (abr) {
            additionalInfo = `${abr}kbps`;
            if (abr >= 320) {
                qualityBadge = '<span class="badge bg-success">高品质</span>';
            } else if (abr >= 192) {
                qualityBadge = '<span class="badge bg-info">标准</span>';
            } else {
                qualityBadge = '<span class="badge bg-secondary">低品质</span>';
            }
        }
    }
    
    div.innerHTML = `
        <div class="d-flex justify-content-between align-items-center">
            <div style="flex: 1;">
                <div class="format-quality d-flex align-items-center gap-2">
                    <span>${qualityText} (${extText})</span>
                    ${qualityBadge}
                </div>
                <div class="format-size text-muted small">
                    ${sizeText}${additionalInfo ? ' • ' + additionalInfo : ''}
                </div>
            </div>
            <button class="btn btn-outline-primary btn-sm" onclick="startDownload('${format.format_id}', '${type}')">
                <i class="bi bi-download"></i>
                下载
            </button>
        </div>
    `;
    
    return div;
}

// 显示字幕选项
function displaySubtitles(subtitles) {
    const subtitleFormats = document.getElementById('subtitleFormats');
    subtitleFormats.innerHTML = '';
    
    if (subtitles.length === 0) {
        subtitleFormats.innerHTML = '<p class="text-muted">无可用字幕</p>';
    } else {
        subtitles.forEach(subtitle => {
            const button = document.createElement('button');
            button.className = 'btn btn-outline-secondary btn-sm subtitle-btn';
            button.innerHTML = `<i class="bi bi-chat-square-text"></i> ${subtitle.language}`;
            button.onclick = () => downloadSubtitle(subtitle.language);
            subtitleFormats.appendChild(button);
        });
    }
}

// 开始下载
async function startDownload(formatId, type) {
    if (!currentVideoInfo) {
        showError('请先解析视频信息');
        return;
    }
    
    try {
        const response = await fetch('/download', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                url: currentVideoInfo.url,
                format_id: formatId,
                type: type
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            const taskId = data.task_id;
            addDownloadTask(taskId, currentVideoInfo.title, type);
            showProgressModal(taskId);
            pollProgress(taskId);
        } else {
            showError(data.error || '下载启动失败');
        }
    } catch (error) {
        showError('网络错误，请重试');
        console.error('Download error:', error);
    }
}

// 下载字幕
async function downloadSubtitle(language) {
    if (!currentVideoInfo) {
        showError('请先解析视频信息');
        return;
    }
    
    try {
        const response = await fetch('/subtitle_download', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                url: currentVideoInfo.url,
                language: language
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // 创建临时下载链接
            const link = document.createElement('a');
            link.href = data.download_url;
            link.download = `${currentVideoInfo.title}_${language}.srt`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            showSuccess('字幕下载成功');
        } else {
            showError(data.error || '字幕下载失败');
        }
    } catch (error) {
        showError('网络错误，请重试');
        console.error('Subtitle download error:', error);
    }
}

// 添加下载任务到侧边栏
function addDownloadTask(taskId, title, type) {
    downloadTasks[taskId] = {
        title: title,
        type: type,
        status: 'starting'
    };
    
    updateDownloadTasksList();
}

// 更新下载任务列表
function updateDownloadTasksList() {
    const container = document.getElementById('downloadTasks');
    
    if (Object.keys(downloadTasks).length === 0) {
        container.innerHTML = '<p class="text-muted">暂无下载任务</p>';
        return;
    }
    
    let html = '';
    Object.entries(downloadTasks).forEach(([taskId, task]) => {
        const statusClass = task.status === 'completed' ? 'completed' : 
                           task.status === 'error' ? 'error' : 'downloading';
        
        const statusIcon = task.status === 'completed' ? 'check-circle' :
                          task.status === 'error' ? 'x-circle' : 'download';
        
        const statusText = task.status === 'completed' ? '已完成' :
                          task.status === 'error' ? '失败' : '下载中';
        
        html += `
            <div class="download-task ${statusClass}">
                <div class="d-flex justify-content-between align-items-start">
                    <div style="flex: 1; min-width: 0;">
                        <div class="fw-bold small text-truncate" title="${task.title}">
                            ${task.title}
                        </div>
                        <div class="text-muted small">
                            <i class="bi bi-${statusIcon}"></i>
                            ${statusText} • ${task.type === 'video' ? '视频' : '音频'}
                        </div>
                        ${task.progress ? `
                            <div class="progress mt-1" style="height: 4px;">
                                <div class="progress-bar" style="width: ${task.progress}%"></div>
                            </div>
                        ` : ''}
                    </div>
                    ${task.downloadUrl ? `
                        <a href="${task.downloadUrl}" class="btn btn-sm btn-outline-success ms-2" download>
                            <i class="bi bi-download"></i>
                        </a>
                    ` : ''}
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// 显示进度模态框
function showProgressModal(taskId) {
    const modal = new bootstrap.Modal(document.getElementById('progressModal'));
    modal.show();
    
    // 重置模态框状态
    document.getElementById('progressFileName').textContent = '准备下载...';
    document.getElementById('progressPercent').textContent = '0%';
    document.getElementById('progressBar').style.width = '0%';
    document.getElementById('progressSpeed').textContent = '-';
    document.getElementById('progressETA').textContent = '-';
    document.getElementById('progressStatus').textContent = '准备中';
    document.getElementById('downloadComplete').style.display = 'none';
    document.getElementById('downloadError').style.display = 'none';
}

// 轮询下载进度
async function pollProgress(taskId) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/progress/${taskId}`);
            const data = await response.json();
            
            if (data.status === 'not_found') {
                clearInterval(pollInterval);
                return;
            }
            
            updateProgressDisplay(data);
            updateDownloadTaskProgress(taskId, data);
            
            if (data.status === 'completed' || data.status === 'error') {
                clearInterval(pollInterval);
                
                if (data.status === 'completed') {
                    showDownloadComplete(data.download_url, data.filename);
                    downloadTasks[taskId].status = 'completed';
                    downloadTasks[taskId].downloadUrl = data.download_url;
                } else {
                    showDownloadError(data.error);
                    downloadTasks[taskId].status = 'error';
                }
                
                updateDownloadTasksList();
            }
        } catch (error) {
            console.error('Progress polling error:', error);
        }
    }, 1000);
}

// 更新进度显示
function updateProgressDisplay(data) {
    document.getElementById('progressPercent').textContent = `${data.percent || 0}%`;
    document.getElementById('progressBar').style.width = `${data.percent || 0}%`;
    
    if (data.speed) {
        document.getElementById('progressSpeed').textContent = formatSpeed(data.speed);
    }
    
    if (data.eta) {
        document.getElementById('progressETA').textContent = formatTime(data.eta);
    }
    
    const statusText = {
        'starting': '准备中',
        'downloading': '下载中',
        'finished': '完成',
        'completed': '已完成',
        'error': '错误'
    };
    
    document.getElementById('progressStatus').textContent = statusText[data.status] || data.status;
}

// 更新任务进度
function updateDownloadTaskProgress(taskId, data) {
    if (downloadTasks[taskId]) {
        downloadTasks[taskId].progress = data.percent;
        downloadTasks[taskId].status = data.status;
        updateDownloadTasksList();
    }
}

// 显示下载完成
function showDownloadComplete(downloadUrl, filename) {
    document.getElementById('downloadComplete').style.display = 'block';
    const downloadLink = document.getElementById('downloadLink');
    downloadLink.href = downloadUrl;
    downloadLink.download = filename;
}

// 显示下载错误
function showDownloadError(error) {
    document.getElementById('downloadError').style.display = 'block';
    document.getElementById('downloadErrorMessage').textContent = error;
}

// 调试格式功能
async function debugFormats(url) {
    try {
        const response = await fetch('/debug_formats', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            console.log('所有格式信息:', data);
            
            // 创建调试信息显示
            let debugInfo = `=== ${data.title} ===\n`;
            debugInfo += `总格式数量: ${data.total_formats}\n\n`;
            
            data.all_formats.forEach((fmt, index) => {
                debugInfo += `格式 ${index + 1}:\n`;
                debugInfo += `  ID: ${fmt.format_id}\n`;
                debugInfo += `  扩展名: ${fmt.ext}\n`;
                debugInfo += `  分辨率: ${fmt.resolution || 'N/A'} (${fmt.width}x${fmt.height})\n`;
                debugInfo += `  视频编码: ${fmt.vcodec}\n`;
                debugInfo += `  音频编码: ${fmt.acodec}\n`;
                debugInfo += `  质量标注: ${fmt.quality}\n`;
                debugInfo += `  文件大小: ${fmt.filesize ? (fmt.filesize / 1024 / 1024).toFixed(2) + ' MB' : 'N/A'}\n`;
                debugInfo += `  FPS: ${fmt.fps || 'N/A'}\n`;
                debugInfo += `  协议: ${fmt.protocol}\n`;
                debugInfo += `  备注: ${fmt.format_note}\n`;
                debugInfo += '\n';
            });
            
            // 显示在新窗口或控制台
            const debugWindow = window.open('', '_blank', 'width=800,height=600');
            debugWindow.document.write(`
                <html>
                <head><title>格式调试信息</title></head>
                <body style="font-family: monospace; white-space: pre-wrap; padding: 20px;">
                    ${debugInfo}
                </body>
                </html>
            `);
            
        } else {
            console.error('调试失败:', data.error);
            alert('调试失败: ' + data.error);
        }
    } catch (error) {
        console.error('调试请求失败:', error);
        alert('调试请求失败');
    }
}

// 工具函数
function showLoading() {
    document.getElementById('loadingState').style.display = 'block';
    document.getElementById('videoInfo').style.display = 'none';
    document.getElementById('welcomeInfo').style.display = 'none';
}

function hideLoading() {
    document.getElementById('loadingState').style.display = 'none';
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('errorInfo').style.display = 'block';
    document.getElementById('videoInfo').style.display = 'none';
    document.getElementById('welcomeInfo').style.display = 'none';
}

function hideError() {
    document.getElementById('errorInfo').style.display = 'none';
}

function hideWelcome() {
    document.getElementById('welcomeInfo').style.display = 'none';
}

function showSuccess(message) {
    // 简单的成功提示，可以用 toast 或其他方式
    console.log('Success:', message);
    // 这里可以实现一个临时的成功提示
}

function formatNumber(num) {
    if (!num) return '0';
    return num.toLocaleString();
}

function formatDuration(seconds) {
    if (!seconds) return '00:00';
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
        return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    } else {
        return `${minutes}:${secs.toString().padStart(2, '0')}`;
    }
}

function formatDate(dateStr) {
    if (!dateStr) return '';
    
    const year = dateStr.substring(0, 4);
    const month = dateStr.substring(4, 6);
    const day = dateStr.substring(6, 8);
    
    return `${year}-${month}-${day}`;
}

function formatSpeed(bytesPerSec) {
    if (!bytesPerSec) return '-';
    
    const units = ['B/s', 'KB/s', 'MB/s', 'GB/s'];
    let size = bytesPerSec;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
}

function formatTime(seconds) {
    if (!seconds) return '-';
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
}
