import numpy
import matplotlib.pyplot as plt
import math

# import csv

clock = 1  # base unit time
warm_up = 50  # number of ignored outcomes

# for report pre
phase = 0
turn = 0
all_customer_1 = 0
total_done_1 = 0
time_wait_1 = 0
customer_in_queue_1 = 0

# for report main
all_customer_m = 0
total_done_m = 0
customer_in_queue_m = 0
turn_m = 0
phase_m = 0


class Customer:
    def __init__(self, arrival_time, service_start_time, mu):
        self.arrival_time = arrival_time
        self.service_start_time = service_start_time
        self.service_time = abs(numpy.random.exponential(1 / mu))
        self.service_end = -1
        self.service_wait = -1

    def end(self, time):
        self.service_end = time
        self.service_wait = time - self.arrival_time - self.service_time


class PreProcessor1:
    def __init__(self, lamb, k):
        self.lamb = lamb
        self.queue = [None] * k
        self.k = k

    def simulation(self, time_passed, time):
        global phase, turn, all_customer_1, total_done_1, time_wait_1, customer_in_queue_1, clock
        # done tasks
        done = []

        c_count = 0  # customers count
        pre_all_customer_1 = all_customer_1
        pre_wait = time_wait_1
        pre_q = customer_in_queue_1
        pre_turn = turn
        for c in self.queue:
            if c is not None:
                c_count = c_count + 1

        if phase > warm_up:
            customer_in_queue_1 = customer_in_queue_1 + c_count  # for average # of customer in queue
            turn = turn + 1

        if c_count == 0:
            # manage the queue
            time = time + time_passed
            next_customers = numpy.random.poisson(self.lamb)
            if phase > warm_up:
                all_customer_1 = all_customer_1 + next_customers  # for average blocks
            if next_customers > self.k:
                for i in range(self.k):
                    self.queue[i] = Customer(time, -1, 5)
            else:
                for i in range(next_customers):
                    self.queue[i] = Customer(time, -1, 5)
        else:
            # processing time
            p_time = 0

            while clock - p_time > 0:
                # break when there is no customer in the queue
                if c_count == 0:
                    break

                # find shortest customer task
                min_s_time = 1000
                for c in self.queue:
                    if c is not None:
                        if c.service_time < min_s_time:
                            min_s_time = c.service_time
                            c_min_index = self.queue.index(c)

                # process
                if min_s_time + p_time > clock:
                    self.queue[c_min_index].service_time = min_s_time - (clock - p_time)
                    self.queue[c_min_index].service_start_time = time
                    self.queue[c_min_index].service_wait = time - self.queue[c_min_index].arrival_time
                    p_time = clock
                else:
                    if self.queue[c_min_index].service_start_time == -1:
                        self.queue[c_min_index].service_start_time = time
                    self.queue[c_min_index].service_end = time + time_passed
                    self.queue[c_min_index].service_wait = time - self.queue[c_min_index].arrival_time
                    if phase > warm_up:
                        time_wait_1 = time_wait_1 + self.queue[c_min_index].service_wait  # for average blocks
                        total_done_1 = total_done_1 + 1  # for average wait time
                    done.append(self.queue[c_min_index])
                    self.queue[c_min_index] = None
                    c_count = c_count - 1
                    p_time = p_time + min_s_time
                    phase = phase + 1  # similar to done processes

            # manage the queue
            time = time + time_passed
            if len(self.queue) == 0:
                next_customers = numpy.random.poisson(self.lamb)
                if phase > warm_up:
                    all_customer_1 = all_customer_1 + next_customers  # for average blocks
                if next_customers > self.k:
                    for i in range(self.k):
                        self.queue[i] = Customer(time, -1, 5)
                else:
                    for i in range(next_customers):
                        self.queue[i] = Customer(time, -1, 5)
            else:
                next_customers = numpy.random.poisson(self.lamb)
                if phase > warm_up:
                    all_customer_1 = all_customer_1 + next_customers  # for average blocks
                if next_customers >= self.k - c_count:
                    for c in self.queue:
                        if c is None:
                            self.queue[self.queue.index(c)] = Customer(time, -1, 5)
                else:
                    temp = next_customers  # fill only #next_customers customers in the queue
                    for c in self.queue:
                        if temp == 0:
                            break
                        else:
                            if c is None:
                                self.queue[self.queue.index(c)] = Customer(time, -1, 5)
                                temp = temp - 1
        wait_1 = time_wait_1 - pre_wait
        customer_1 = all_customer_1 - pre_all_customer_1
        q = customer_in_queue_1 - pre_q
        t = turn - pre_turn
        return done, customer_1, wait_1, q, t


class PreProcessor2:
    def __init__(self, lamb, k):
        self.lamb = lamb
        self.queue = [None] * k
        self.k = k

    def simulation(self, time_passed, time):
        # done tasks
        done = []

        c_count = 0  # customers count
        for c in self.queue:
            if c is not None:
                c_count = c_count + 1

        if c_count == 0:
            # manage the queue
            time = time + time_passed
            next_customers = numpy.random.poisson(self.lamb)
            if next_customers > self.k:
                for i in range(self.k):
                    self.queue[i] = Customer(time, -1, 3)
            else:
                for i in range(next_customers):
                    self.queue[i] = Customer(time, -1, 3)
        else:
            # processing time
            p_time = 0
            while clock - p_time > 0:
                # break when there is no customer in the queue
                if c_count == 0:
                    break

                # find a random not none index
                found = False
                while not found:
                    index = numpy.random.randint(0, self.k)
                    if self.queue[index] is not None:
                        found = True

                # process
                if self.queue[index].service_time + p_time > clock:
                    self.queue[index].service_time = self.queue[index].service_time - (clock - p_time)
                    self.queue[index].service_start_time = time
                    self.queue[index].service_wait = time - self.queue[index].arrival_time
                    p_time = 1
                else:
                    if self.queue[index].service_start_time == -1:
                        self.queue[index].service_start_time = time
                    self.queue[index].service_end = time + time_passed
                    self.queue[index].service_wait = time - self.queue[index].arrival_time
                    done.append(self.queue[index])
                    p_time = p_time + self.queue[index].service_time
                    self.queue[index] = None
                    c_count = c_count - 1

            # manage the queue
            time = time + time_passed
            if len(self.queue) == 0:
                next_customers = numpy.random.poisson(self.lamb)
                if next_customers > self.k:
                    for i in range(self.k):
                        self.queue[i] = Customer(time, -1, 3)
                else:
                    for i in range(next_customers):
                        self.queue[i] = Customer(time, -1, 3)
            else:
                next_customers = numpy.random.poisson(self.lamb)
                if next_customers >= self.k - c_count:
                    for c in self.queue:
                        if c is None:
                            self.queue[self.queue.index(c)] = Customer(time, -1, 3)
                else:
                    temp = next_customers  # fill only #next_customers customers in the queue
                    for c in self.queue:
                        if temp == 0:
                            break
                        else:
                            if c is None:
                                self.queue[self.queue.index(c)] = Customer(time, -1, 3)
                                temp = temp - 1

        return done


class MainProcessor:

    def __init__(self, k):
        self.queue = [None] * k
        self.k = k
        self.done = []

    def simulation(self, time_passed, time, next_customers):
        global phase_m, all_customer_m, total_done_m, customer_in_queue_m, turn_m
        all_customer_m = all_customer_m + len(next_customers)  # for calculating all arrival customers
        c_count = 0  # customers count
        tot_time = []  # for calculating total time of a customer in system

        pre_done = total_done_m
        pre_q = customer_in_queue_m
        pre_turn = turn_m
        for c in self.queue:
            if c is not None:
                c_count = c_count + 1

        if phase_m > warm_up:
            customer_in_queue_m = customer_in_queue_m + c_count
            turn_m = turn_m + 1

        if len(next_customers) == 0 and c_count == 0:
            time = time + time_passed

        else:
            # put new arrival in main processor queue (left-overs dropped!)
            for c_a in next_customers:
                for c in self.queue:
                    if c is None:
                        c_a.service_time = abs(numpy.random.exponential(1 / 1))
                        self.queue[self.queue.index(c)] = c_a
                        c_count = c_count + 1
                        break
                if c_count == self.k:
                    break

            # process
            for c in self.queue:
                if c is not None:
                    if c.service_time - (clock / c_count) <= 0:
                        index = self.queue.index(c)
                        if self.queue[index].service_start_time == -1:
                            self.queue[index].service_start_time = time
                        self.queue[index].service_end = time + time_passed
                        self.done.append(c)
                        tot_time.append(c)  # customer done in one clock
                        self.queue[index] = None
                        phase_m = phase_m + 1
                        if phase > warm_up:
                            total_done_m = total_done_m + 1  # for block chance

                    else:
                        index = self.queue.index(c)
                        self.queue[index].service_time = self.queue[index].service_time - (clock / c_count)
                        self.queue[index].service_start_time = time
        now_done = total_done_m - pre_done
        q = customer_in_queue_m - pre_q
        t = turn_m - pre_turn
        return len(next_customers), now_done, q, t, tot_time


class MainProcessorExtra:

    def __init__(self, k):
        self.queue = [None] * k
        self.k = k
        self.done = []

    def simulation(self, time_passed, time, next_customers):
        global phase_m, all_customer_m, total_done_m, customer_in_queue_m, turn_m
        all_customer_m = all_customer_m + len(next_customers)  # for calculating all arrival customers
        c_count = 0  # customers count
        for c in self.queue:
            if c is not None:
                c_count = c_count + 1

        if phase_m > warm_up:
            customer_in_queue_m = customer_in_queue_m + c_count
            turn_m = turn_m + 1

        if len(next_customers) == 0 and c_count == 0:
            time = time + time_passed

        else:
            # put new arrival in main processor queue (left-overs dropped!)
            for c_a in next_customers:
                for c in self.queue:
                    if c is None:
                        c_a.service_time = abs(numpy.random.exponential(1 / 1))
                        self.queue[self.queue.index(c)] = c_a
                        c_count = c_count + 1
                        break
                if c_count == self.k:
                    break

            # process section
            p_time = 0  # processing time
            while clock - p_time > 0:
                # break when there is no customer in the queue
                if c_count == 0:
                    break

                # find shortest customer arrival time
                min_a_time = 9999999999999
                for c in self.queue:
                    if c is not None:
                        if c.arrival_time < min_a_time:
                            min_a_time = c.arrival_time
                            c_min_index = self.queue.index(c)

                # process
                if self.queue[c_min_index].service_time + p_time > clock:
                    self.queue[c_min_index].service_time = self.queue[c_min_index].service_time - (clock - p_time)
                    self.queue[c_min_index].service_start_time = time
                    self.queue[c_min_index].service_wait = time - self.queue[c_min_index].arrival_time
                    p_time = clock
                else:
                    if self.queue[c_min_index].service_start_time == -1:
                        self.queue[c_min_index].service_start_time = time
                    self.queue[c_min_index].service_end = time + time_passed
                    self.queue[c_min_index].service_wait = time - self.queue[c_min_index].arrival_time
                    self.done.append(self.queue[c_min_index])
                    p_time = p_time + self.queue[c_min_index].service_time
                    self.queue[c_min_index] = None
                    phase_m = phase_m + 1
                    if phase_m > warm_up:
                        total_done_m = total_done_m + 1  # for block chance
                    c_count = c_count - 1
                time = time + time_passed

        return None


def processor_sharing():
    global phase_m, all_customer_m, total_done_m, customer_in_queue_m, turn_m, phase, turn, all_customer_1, total_done_1, \
        time_wait_1, customer_in_queue_1, clock

    phase = 0
    turn = 0
    all_customer_1 = 0
    total_done_1 = 0
    time_wait_1 = 0
    customer_in_queue_1 = 0

    all_customer_m = 0
    total_done_m = 0
    customer_in_queue_m = 0
    turn_m = 0
    phase_m = 0

    # Main Section
    ans = dict()

    for k in range(8, 17):
        print('*** Main Section Started *** \n')
        pp1 = PreProcessor1(7, 100)
        pp2 = PreProcessor2(2, 12)
        mp = MainProcessor(8)
        time = 0
        simulation_times = 5000000
        prec_val = make_dict_for_data()
        simulation_R = 1
        all_done = [0] * 6
        while len(mp.done) < simulation_times:
            output1, all_customer_per_simulation_1, time_wait_per_simulation_1, \
            customer_in_queue_per_simulation_1, turn_1_per_simulation = pp1.simulation(
                clock, time)

            output2 = pp2.simulation(clock, time)
            output = output1 + output2

            all_customer_per_simulation_m, done_customer_per_simulation_m, \
            customer_in_queue_per_simulation_m, turn_m_per_simulation, tot_m = mp.simulation(
                clock, time, output)

            if phase > warm_up:
                prec_val['a'].append(calc_pb(all_customer_per_simulation_1, len(output1)))
                prec_val['b'].append(calc_lq(customer_in_queue_per_simulation_1, turn_1_per_simulation))
                prec_val['c'].append(calc_wq(time_wait_per_simulation_1, len(output1)))
                prec_val['d'].append(calc_pb(all_customer_per_simulation_m, done_customer_per_simulation_m))
                prec_val['e'].append(calc_tot_time(tot_m))
                prec_val['f'].append(calc_lq(customer_in_queue_per_simulation_m, turn_m_per_simulation))
                # all_precs = calc_all_precisions(simulation_R, prec_val)
                simulation_R = simulation_R + 1
                '''
                if all_precs['a'] > 0 and all_precs['a'] < 0.05 and all_done[0] == 0:
                    print("reached pb1 precision in:", simulation_R - 1)
                    all_done[0] = 1
                if all_precs['b'] > 0 and all_precs['b'] < 0.05 and all_done[1] == 0:
                    print("reached lq1 precision in:", simulation_R - 1)
                    all_done[1] = 1
                if all_precs['c'] > 0 and all_precs['c'] < 0.05 and all_done[2] == 0:
                    print("reached wq1 precision in:", simulation_R - 1)
                    all_done[2] = 1
                if all_precs['d'] > 0 and all_precs['d'] < 0.05 and all_done[3] == 0:
                    print("reached pb3 precision in:", simulation_R - 1)
                    all_done[3] = 1
                if all_precs['e'] > 0 and all_precs['e'] < 0.05 and all_done[4] == 0:
                    print("reached total time precision in:", simulation_R - 1)
                    all_done[4] = 1
                if all_precs['f'] > 0 and all_precs['f'] < 0.05 and all_done[5] == 0:
                    print("reached lq3 precision in:", simulation_R - 1)
                    all_done[5] = 1

                if sum(all_done) == 6:
                    print("reached precision for all results in:", simulation_R - 1)
                    all_done[0] = 8
                    print("1.1. PB1 : ")
                    answer = prec_val['a'][-1]
                    answer = '%' + str(answer)
                    print(answer)
                    print("1.1 PB1 precision : ", all_precs['a'])
                    print()

                    print("1.2. LQ1 : ")
                    answer = prec_val['b'][-1]
                    answer = str(answer)
                    print(answer)
                    print("1.2. LQ1 precision : ", all_precs['b'])

                    print("1.3. WQ1 : ")
                    answer = prec_val['c'][-1]
                    answer = str(answer)
                    print(answer)
                    print("1.3. WQ1 precision : ", all_precs['c'])

                    print("2.1. PB3 : ")
                    answer = prec_val['d'][-1]
                    answer = str(answer)
                    print(answer)
                    print("2.1. PB3 precision : ", all_precs['d'])
                    ans[k].append(answer)

                    mean__time = prec_val['e'][-1]
                    print("2.2. Total Times : ")
                    print(mean__time)
                    print("2.2. Total Times : ", all_precs['e'])
                    ans[k].append(mean__time)

                    print("2.3. LQ3 : ")
                    answer = prec_val['f'][-1]
                    answer = str(answer)
                    print(answer)
                    print("2.3. LQ3 : ", all_precs['f'])
                    ans[k].append(answer)

                    print("===============")
                    print()
                '''
            time = time + clock

        p = calc_all_precisions(simulation_R - 1, prec_val)
        # calculate summary statistics
        print('General Statistics In PS : ')
        print('K = ' + str(k))
        print('Simulation Rounds = ' + str(simulation_R))
        ans[k] = []  # for plot

        print()
        answer = calc_pb(all_customer_1, total_done_1)
        answer = '%' + str(answer)
        print("1.1. PB1 : ", answer)
        print("1.1 PB1 precision : ", p['a'])
        print()


        answer = calc_lq(customer_in_queue_1, turn)
        answer = str(answer)
        print("1.2. LQ1 : ", answer)
        print("1.2. LQ1 precision : ", p['b'])
        print()

        answer = calc_wq(time_wait_1, total_done_1)
        answer = str(answer)
        print("1.3. WQ1 : ", answer)
        print("1.3. WQ1 precision : ", p['c'])
        print()

        answer = calc_pb(all_customer_m, total_done_m)
        answer = '%' + str(answer)
        print("2.1. PB3 : ", answer, "%")
        print("2.1. PB3 precision : ", p['d'])
        ans[k].append(answer)
        print()

        warmed_up = mp.done[warm_up:]
        mean__time = calc_tot_time(warmed_up)
        print("2.2. Total Times : ", mean__time)
        print("2.2. Total Times : ", p['e'])
        ans[k].append(mean__time)
        print()

        answer = calc_lq(customer_in_queue_m, turn_m)
        answer = str(answer)
        print("2.3. LQ3 : ", answer)
        print("2.3. LQ3 : ", p['f'])
        ans[k].append(answer)
        print()

        print("===============")
        print()

    '''
        # write output full data set to csv
        outfile = open('system_output_%s_simulations.csv' % simulation_times, 'w')
        output = csv.writer(outfile)
        output.writerow(
            ['Customer', 'Arrival_Time', 'Service_Start_Time', 'Service_Time', 'Service_Wait', 'Service_End'])
        i = 0
        for customer in mp.done:
            i = i + 1
            outrow = []
            outrow.append(i)
            outrow.append(customer.arrival_time)
            outrow.append(customer.service_start_time)
            outrow.append(customer.service_time)
            outrow.append(customer.service_wait)
            outrow.append(customer.service_end)
            output.writerow(outrow)
        outfile.close()
    '''

    show_plots(ans)


def make_dict_for_data():
    res = dict()
    res['a'] = list()
    res['b'] = list()
    res['c'] = list()
    res['d'] = list()
    res['e'] = list()
    res['f'] = list()
    return res


def compute_precision(data_in, r_in):
    mean = sum(data_in) / len(data_in)
    std = math.sqrt(sum((x - mean) ** 2 for x in data_in) / len(data_in))
    if r_in == 0 or mean == 0:
        return 0
    prec = 1.96 * std * (1 / (math.sqrt(r_in) * mean))

    return prec


def calc_pb(all_customer, total_done):
    if all_customer == 0:
        return 0
    pb_res = ((all_customer - total_done) / all_customer) * 100

    return pb_res


def calc_lq(customer_in_queue, turn_in):
    if turn_in == 0:
        return 0
    lq_res = customer_in_queue / turn_in
    return lq_res


def calc_wq(time_wait, total_done):
    if total_done == 0:
        return 0
    wq_res = time_wait / total_done

    return wq_res


def calc_tot_time(warm):
    total_times = [c.service_wait + c.service_time for c in warm]
    if total_times == 0 or len(total_times) == 0:
        return 0
    mean_time = sum(total_times) / len(total_times)
    return mean_time


def calc_all_precisions(Rin, pre_data):
    res = dict()
    res['a'] = compute_precision(pre_data['a'], Rin)
    res['b'] = compute_precision(pre_data['b'], Rin)
    res['c'] = compute_precision(pre_data['c'], Rin)
    res['d'] = compute_precision(pre_data['d'], Rin)
    res['e'] = compute_precision(pre_data['e'], Rin)
    res['f'] = compute_precision(pre_data['f'], Rin)
    return res


def show_plots(data_in):
    pb3, tot_time, lq3 = convert_dict_values(data_in)
    data_key = data_in.keys()
    plt.bar(data_key, pb3, color="green", align='center', alpha=0.5)
    plt.ylabel('PB3')
    plt.xlabel('K')
    plt.title('Probability of Blocking customer in Main Server')
    plt.show()

    plt.bar(data_key, tot_time, color="red", align='center', alpha=0.5)
    plt.ylabel('Total Time')
    plt.xlabel('K')
    plt.title('Average Total Process Time in Main Server')
    plt.show()

    plt.bar(data_key, lq3, align='center', alpha=0.5)
    plt.ylabel('LQ3')
    plt.xlabel('K')
    plt.title('Average Process Queue Size in Main Server')
    plt.show()


def convert_dict_values(dict_in):
    # PB3
    pb3_out = list()
    tot_time_out = list()
    lq3_out = list()
    for item in dict_in.keys():
        pb3_out.append(dict_in[item][0])
        tot_time_out.append(dict_in[item][1])
        lq3_out.append(dict_in[item][2])

    return pb3_out, tot_time_out, lq3_out


def first_come_first_served():
    global phase_m, all_customer_m, total_done_m, customer_in_queue_m, turn_m, phase, turn, all_customer_1, total_done_1, \
        time_wait_1, customer_in_queue_1, clock

    phase = 0
    turn = 0
    all_customer_1 = 0
    total_done_1 = 0
    time_wait_1 = 0
    customer_in_queue_1 = 0

    all_customer_m = 0
    total_done_m = 0
    customer_in_queue_m = 0
    turn_m = 0
    phase_m = 0

    # Extra Section
    ans = dict()
    for k in range(8, 17):
        print('*** Extra Section Started *** \n')
        pp1 = PreProcessor1(7, 100)
        pp2 = PreProcessor2(2, 12)
        mp = MainProcessorExtra(k)
        time = 0
        simulation_times = 50000
        while len(mp.done) < simulation_times:
            output1 = pp1.simulation(clock, time)
            output2 = pp2.simulation(clock, time)
            output = output1 + output2
            mp.simulation(clock, time, output)
            time = time + clock

        # calculate summary statistics
        print('General Statistics In FCFS : ')
        print('K = ' + str(k))

        ans[k] = []  # for plot

        print("1.1. PB1 : ")
        answer = ((all_customer_1 - total_done_1) / all_customer_1) * 100
        answer = '%' + str(answer)
        print(answer)

        print("1.2. LQ1 : ")
        answer = (customer_in_queue_1 / turn)
        answer = str(answer)
        print(answer)

        print("1.3. WQ1 : ")
        answer = (time_wait_1 / total_done_1)
        answer = str(answer)
        print(answer)

        print("2.1. PB3 : ")
        answer = ((all_customer_m - total_done_m) / all_customer_m) * 100
        answer = str(answer)
        print(answer)
        ans[k].append(answer)

        warmed_up = mp.done[warm_up:]
        total__times = [c.service_wait + c.service_time for c in warmed_up]
        mean__time = sum(total__times) / len(total__times)
        print("2.2. Total Times : ")
        print(mean__time)
        ans[k].append(mean__time)

        print("2.3. LQ3 : ")
        answer = (customer_in_queue_m / turn_m)
        answer = str(answer)
        print(answer)
        ans[k].append(answer)

        print("===============")
        print()

        '''
        # write output full data set to csv
        outfile = open('system_output_%s_simulations_extra.csv' % simulation_times, 'w')
        output = csv.writer(outfile)
        output.writerow(
            ['Customer', 'Arrival_Time', 'Service_Start_Time', 'Service_Time', 'Service_Wait', 'Service_End'])
        i = 0
        for customer in mp.done:
            i = i + 1
            outrow = []
            outrow.append(i)
            outrow.append(customer.arrival_time)
            outrow.append(customer.service_start_time)
            outrow.append(customer.service_time)
            outrow.append(customer.service_wait)
            outrow.append(customer.service_end)
            output.writerow(outrow)
        outfile.close()
        '''


if __name__ == '__main__':
    Extra = False
    Normal = True
    if Normal:
        processor_sharing()
    if Extra:
        first_come_first_served()
