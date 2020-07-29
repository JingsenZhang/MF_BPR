import subprocess
import threading
import time
try:
    import queue
except ImportError:
    import Queue as queue


def execute_command(cmd):
    print('begin command: ', cmd)
    start = time.time()
    s = subprocess.Popen(str(cmd), stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    stderrinfo, stdoutinfo = s.communicate()
    line = stderrinfo.rstrip().decode('utf8')
    print(cmd, line[line.find('final best performance'):])
    #print('end command: ', cmd, 'time cost: ', str(time.time() - start))


def consume(q):
    while (True):
        name = threading.currentThread().getName()
        #print("Thread: {0} start get item from queue[current size = {1}] at time = {2} \n".format(name, q.qsize(), time.strftime('%H:%M:%S')))
        cmd = q.get()
        execute_command(cmd)
        time.sleep(3)
        #print("Thread: {0} finish process item from queue[current size = {1}] at time = {2} \n".format(name, q.qsize(),time.strftime('%H:%M:%S')))
        q.task_done()


def producer(q, cmd_list):
    for i in range(len(cmd_list)):
        name = threading.currentThread().getName()
        #print("Thread: {0} start put item into queue[current size = {1}] at time = {2} \n".format(name, q.qsize(), time.strftime('%H:%M:%S')))
        item = cmd_list[i]
        q.put(item)
        #print("Thread: {0} successfully put item into queue[current size = {1}] at time = {2} \n".format(name, q.qsize(), time.strftime('%H:%M:%S')))
    q.join()




if __name__ == '__main__':
    cmd_list = []
    amazon_paths = ["./data/amazon/app/", "./data/amazon/phone/", "./data/amazon/pet/",
                    "./data/amazon/baby/", "./data/amazon/instant/", "./data/amazon/Digital/"]
    for baseline in ['BPR.py', 'NCF.py', 'PMF.py']:
        for path in amazon_paths:
            cmd = 'python3 ' + baseline + ' --data_path ' + path
            cmd_list.append(cmd)


    threads_num = 4
    q = queue.Queue(maxsize=threads_num)
    for i in range(threads_num):
        t = threading.Thread(name="ConsumerThread-" + str(i), target=consume, args=(q,))
        t.start()

    t = threading.Thread(name="ProducerThread", target=producer, args=(q, cmd_list,))
    t.start()

    q.join()













