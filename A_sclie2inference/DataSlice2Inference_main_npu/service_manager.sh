#!/bin/bash
# RT-DETR 服务管理脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="rtdetr-processor"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
PYTHON_SCRIPT="${SCRIPT_DIR}/service_main.py"
PID_FILE="/var/run/${SERVICE_NAME}.pid"
LOG_FILE="${SCRIPT_DIR}/logs/service.log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

check_dependencies() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装"
        exit 1
    fi
    
    if ! python3 -c "import watchdog" 2>/dev/null; then
        print_error "缺少依赖: watchdog"
        print_status "请运行: pip install -r requirements.txt"
        exit 1
    fi
}

install_service() {
    print_status "安装 systemd 服务..."
    
    if [ ! -f "${SCRIPT_DIR}/systemd/${SERVICE_NAME}.service" ]; then
        print_error "服务文件不存在: ${SCRIPT_DIR}/systemd/${SERVICE_NAME}.service"
        exit 1
    fi
    
    sudo cp "${SCRIPT_DIR}/systemd/${SERVICE_NAME}.service" "${SERVICE_FILE}"
    sudo sed -i "s|@PROJECT_ROOT@|${SCRIPT_DIR}|g" "${SERVICE_FILE}"
    sudo sed -i "s|@PYTHON_PATH@|$(which python3)|g" "${SERVICE_FILE}"
    
    sudo systemctl daemon-reload
    sudo systemctl enable "${SERVICE_NAME}"
    
    print_status "服务已安装并启用"
    print_status "使用以下命令管理服务:"
    echo "  sudo systemctl start ${SERVICE_NAME}    # 启动服务"
    echo "  sudo systemctl stop ${SERVICE_NAME}     # 停止服务"
    echo "  sudo systemctl restart ${SERVICE_NAME}  # 重启服务"
    echo "  sudo systemctl status ${SERVICE_NAME}   # 查看状态"
}

uninstall_service() {
    print_status "卸载 systemd 服务..."
    
    if systemctl is-active --quiet "${SERVICE_NAME}"; then
        sudo systemctl stop "${SERVICE_NAME}"
    fi
    
    if systemctl is-enabled --quiet "${SERVICE_NAME}"; then
        sudo systemctl disable "${SERVICE_NAME}"
    fi
    
    if [ -f "${SERVICE_FILE}" ]; then
        sudo rm "${SERVICE_FILE}"
        sudo systemctl daemon-reload
        print_status "服务已卸载"
    else
        print_warning "服务文件不存在，可能未安装"
    fi
}

start_service() {
    # 检查服务是否已安装
    if [ ! -f "${SERVICE_FILE}" ]; then
        print_error "服务未安装！请先运行: $0 install"
        print_status "或者使用直接启动模式: $0 start-direct"
        exit 1
    fi
    
    if systemctl is-active --quiet "${SERVICE_NAME}"; then
        print_warning "服务已在运行"
        return
    fi
    
    print_status "启动服务..."
    sudo systemctl start "${SERVICE_NAME}"
    sleep 2
    
    if systemctl is-active --quiet "${SERVICE_NAME}"; then
        print_status "服务启动成功"
    else
        print_error "服务启动失败"
        sudo systemctl status "${SERVICE_NAME}"
        exit 1
    fi
}

stop_service() {
    if ! systemctl is-active --quiet "${SERVICE_NAME}"; then
        print_warning "服务未运行"
        return
    fi
    
    print_status "停止服务..."
    sudo systemctl stop "${SERVICE_NAME}"
    print_status "服务已停止"
}

restart_service() {
    print_status "重启服务..."
    sudo systemctl restart "${SERVICE_NAME}"
    sleep 2
    
    if systemctl is-active --quiet "${SERVICE_NAME}"; then
        print_status "服务重启成功"
    else
        print_error "服务重启失败"
        sudo systemctl status "${SERVICE_NAME}"
        exit 1
    fi
}

status_service() {
    sudo systemctl status "${SERVICE_NAME}"
}

logs_service() {
    if [ "$1" == "-f" ] || [ "$1" == "--follow" ]; then
        sudo journalctl -u "${SERVICE_NAME}" -f
    else
        sudo journalctl -u "${SERVICE_NAME}" -n 100 --no-pager
    fi
}

start_direct() {
    print_status "直接启动服务（不使用systemd）..."
    check_dependencies
    
    cd "${SCRIPT_DIR}"
    python3 "${PYTHON_SCRIPT}" --config "${SCRIPT_DIR}/config.yaml"
}

show_help() {
    cat << EOF
RT-DETR 服务管理脚本

用法: $0 <command> [options]

命令:
  install      安装 systemd 服务
  uninstall    卸载 systemd 服务
  start        启动服务
  stop         停止服务
  restart      重启服务
  status       查看服务状态
  logs         查看服务日志
  logs -f      实时查看服务日志
  start-direct 直接启动服务（不使用systemd）
  help         显示此帮助信息

示例:
  $0 install          # 安装服务
  $0 start            # 启动服务
  $0 status           # 查看状态
  $0 logs -f          # 实时查看日志
EOF
}

# 主逻辑
case "${1:-help}" in
    install)
        install_service
        ;;
    uninstall)
        uninstall_service
        ;;
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        status_service
        ;;
    logs)
        logs_service "$2"
        ;;
    start-direct)
        start_direct
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "未知命令: $1"
        show_help
        exit 1
        ;;
esac

