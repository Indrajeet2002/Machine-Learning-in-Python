import numpy as np
import os
import json

import pandas as pd
from pm4py.algo.conformance.tokenreplay.variants.token_replay import *
from pm4py.objects.petri_net import semantics

from pydream.util.DecayFunctions import LinearDecay, REGISTER
from pydream.util.TimedStateSamples import TimedStateSample
from pydream.util.Functions import time_delta_seconds


class EnhancedPN:
    def __init__(self, net, initial_marking, decay_function_file=None):
        """
        Creates a new instance of an enhanced petri net
        :param net: petri net loaded from pm4py
        :param initial_marking: initial marking from pm4py
        :param decay_function_file: default=None, path to existing decay function file for the petri net
        """
        self.config = {}
        self.decay_functions = {}
        self.net = net
        self.initial_marking = initial_marking
        self.resource_keys = None

        if decay_function_file is not None:
            self.loadFromFile(decay_function_file)

        self.activity_key = "concept:name"
        self.timestamp_key = "time:timestamp"
        self.type_key = "type"
        self.MAX_REC_DEPTH = 50

        self.place_list = list()
        for place in self.net.places:
            self.place_list.append(str(place))
        self.place_list = sorted(self.place_list)

        self.trans_map = {}
        for t in self.net.transitions:
            self.trans_map[t.label] = t

    def enhance(self, log_wrapper):
        """
        Enhance a given petri net based on an event log.
        :param log_wrapper: Event log under consideration as LogWrapper
        :return:
        """

        """ Standard Enhancement """
        beta = float(10)

        reactivation_deltas = {}
        for place in self.net.places:
            reactivation_deltas[str(place)] = list()

        log_wrapper.iterator_reset()
        last_activation = {}
        while (log_wrapper.iterator_hasNext()):
            trace = log_wrapper.iterator_next()

            for place in self.net.places:
                if place in self.initial_marking:
                    last_activation[str(place)] = trace[0]['time:timestamp']
                else:
                    last_activation[str(place)] = -1

            """ Replay and estimate parameters """
            places_shortest_path_by_hidden = get_places_shortest_path_by_hidden(self.net, self.MAX_REC_DEPTH)
            marking = copy(self.initial_marking)

            for event in trace:
                if event[self.activity_key] in self.trans_map.keys():
                    activated_places = []
                    toi = self.trans_map[event[self.activity_key]]

                    """ If Transition of interest is not enabled yet, then go through hidden"""
                    if not semantics.is_enabled(toi, self.net, marking):

                        _, _, act_trans, _ = apply_hidden_trans(toi, self.net, copy(marking),
                                                                places_shortest_path_by_hidden, [], 0, set(),
                                                                [copy(marking)])
                        for act_tran in act_trans:
                            for arc in act_tran.out_arcs:
                                activated_places.append(arc.target)
                            marking = semantics.execute(act_tran, self.net, marking)

                    """ If Transition of interest is STILL not enabled yet, then naively add missing token to fulfill firing rule"""
                    if not semantics.is_enabled(toi, self.net, marking):
                        for arc in toi.in_arcs:
                            if arc.source not in marking:
                                marking[arc.source] += 1

                    """ Fire transition of interest """
                    for arc in toi.out_arcs:
                        activated_places.append(arc.target)
                    marking = semantics.execute(toi, self.net, marking)

                    """ Marking is gone - transition could not be fired ..."""
                    if marking is None:
                        raise ValueError("Invalid Marking - Transition " + toi + " could not be fired.")

                    """ Update Time Recordings """
                    for activated_place in activated_places:
                        if last_activation[str(activated_place)] != -1:
                            time_delta = time_delta_seconds(last_activation[str(activated_place)],
                                                            event['time:timestamp'])
                            if time_delta > 0:
                                reactivation_deltas[str(place)].append(time_delta)
                        last_activation[str(activated_place)] = event['time:timestamp']

        """ Calculate decay function parameter """
        for place in self.net.places:
            if len(reactivation_deltas[str(place)]) > 1:
                self.decay_functions[str(place)] = LinearDecay(alpha=1 / np.mean(reactivation_deltas[str(place)]),
                                                               beta=beta)
            else:
                self.decay_functions[str(place)] = LinearDecay(alpha=1 / log_wrapper.max_trace_duration, beta=beta)

        """ Get resource keys to store """
        self.resource_keys = log_wrapper.getResourceKeys()

    def load_severity_scores(self, elix_file, charl_file):
        if os.path.exists(elix_file):
            elix = pd.read_csv(elix_file, index_col=[0])
            elix.loc[elix['index'] == '1-4', 'index'] = 2
            elix.loc[elix['index'] == '>=5', 'index'] = 5
            elix.loc[elix['windex_ahrq'] == '<0', 'windex_ahrq'] = -1
            elix.loc[elix['windex_ahrq'] == '0', 'windex_ahrq'] = 0
            elix.loc[elix['windex_ahrq'] == '1-4', 'windex_ahrq'] = 2
            elix.loc[elix['windex_ahrq'] == '>=5', 'windex_ahrq'] = 5
            elix.loc[elix['windex_vw'] == '<0', 'windex_vw'] = -1
            elix.loc[elix['windex_vw'] == '0', 'windex_vw'] = 0
            elix.loc[elix['windex_vw'] == '1-4', 'windex_vw'] = 2
            elix.loc[elix['windex_vw'] == '>=5', 'windex_vw'] = 5
        else:
            print('[WARNING] : cannot read elix file ', elix_file)

        if os.path.exists(charl_file):
            charl = pd.read_csv(charl_file, index_col=[0])
            charl.loc[charl['index'] == '1-2', 'index'] = 1
            charl.loc[charl['index'] == '3-4', 'index'] = 3
            charl.loc[charl['index'] == '>=5', 'index'] = 5
            charl.loc[charl['windex'] == '1-2', 'windex'] = 1
            charl.loc[charl['windex'] == '3-4', 'windex'] = 3
            charl.loc[charl['windex'] == '>=5', 'windex'] = 5
        else:
            print('[WARNING] : cannot read charl file ', charl_file)

        self.severity = pd.merge(elix, charl, on='patient')

    def get_severity_score(self, patient_id):
        df_list = None
        tmp_df = self.severity[self.severity.patient == int(patient_id)]
        if len(tmp_df) == 1:
            tmp_df = tmp_df.loc[:, tmp_df.columns != 'patient']
            df_list = tmp_df.values[0]
        else:
            if len(tmp_df) == 0:
                print('[WARNING] : severity score for patient:', patient_id, 'not found.')
            else:
                print('[WARNING] : severity score data inconsistent for patient:', patient_id)

        # map(int, df_list)
        return list(map(int, df_list))

    def decay_replay(self, log_wrapper, elix_file, charl_file, resources=None):
        """
        Decay Replay on given event log.
        :param log_wrapper: Input event log as LogWrapper to be replayed.
        :param resources: Resource keys to count (must have been counted during Petri net enhancement already!), as a list
        :return: list of timed state samples as JSON, list of timed state sample objects
        """
        tss = list()
        tss_objs = list()

        decay_values = {}
        token_counts = {}
        marks = {}

        last_activation = {}

        """ Initialize Resource Counter """
        count_resources = False
        resource_counter = None
        if log_wrapper.resource_keys is not None:
            count_resources = True
            resource_counter = dict()
            for key in log_wrapper.resource_keys.keys():
                resource_counter[key] = 0
        """ ---> """

        self.load_severity_scores(elix_file, charl_file)
        log_wrapper.iterator_reset()
        while log_wrapper.iterator_hasNext():
            trace = log_wrapper.iterator_next()

            resource_count = copy(resource_counter)
            """ Reset all counts for the next trace """
            for place in self.net.places:
                if place in self.initial_marking:
                    last_activation[str(place)] = trace[0]['time:timestamp']
                else:
                    last_activation[str(place)] = -1

            for place in self.net.places:
                decay_values[str(place)] = 0.0
                token_counts[str(place)] = 0.0
                marks[str(place)] = 0.0
            """ ----------------------------------> """

            places_shortest_path_by_hidden = get_places_shortest_path_by_hidden(self.net, self.MAX_REC_DEPTH)
            marking = copy(self.initial_marking)

            """ Initialize counts based on initial marking """
            for place in marking:
                decay_values[str(place)] = self.decay_functions[str(place)].decay(t=0)
                token_counts[str(place)] += 1
                marks[str(place)] = 1
            """ ----------------------------------> """

            """ Replay """
            time_past = None
            time_recent = None
            init_time = None
            for event_id in range(len(trace)):
                event = trace[event_id]
                if event_id == 0:
                    init_time = event['time:timestamp']

                time_past = time_recent
                time_recent = event['time:timestamp']

                if event[self.activity_key] in self.trans_map.keys():
                    activated_places = list()

                    toi = self.trans_map[event[self.activity_key]]
                    """ If Transition of interest is not enabled yet, then go through hidden"""
                    if not semantics.is_enabled(toi, self.net, marking):
                        _, _, act_trans, _ = apply_hidden_trans(toi, self.net, copy(marking),
                                                                places_shortest_path_by_hidden, [], 0, set(),
                                                                [copy(marking)])
                        for act_tran in act_trans:
                            for arc in act_tran.out_arcs:
                                activated_places.append(arc.target)
                            marking = semantics.execute(act_tran, self.net, marking)
                    """ If Transition of interest is STILL not enabled yet, then naively add missing token to fulfill firing rule"""
                    if not semantics.is_enabled(toi, self.net, marking):
                        for arc in toi.in_arcs:
                            if arc.source not in marking:
                                marking[arc.source] += 1
                    """ Fire transition of interest """
                    for arc in toi.out_arcs:
                        activated_places.append(arc.target)
                    marking = semantics.execute(toi, self.net, marking)
                    """ Marking is gone - transition could not be fired ..."""
                    if marking is None:
                        raise ValueError("Invalid Marking - Transition " + toi + " could not be fired.")
                    """ ----->"""

                    """ Update Time Recordings """
                    for activated_place in activated_places:
                        last_activation[str(activated_place)] = event['time:timestamp']

                    """ Count Resources"""
                    if count_resources and resources is not None:
                        for resource_key in resources:
                            if resource_key in event.keys():
                                val = resource_key + "_:_" + event[resource_key]
                                if val in resource_count.keys():
                                    resource_count[val] += 1

                    """ Update Vectors and create TimedStateSamples """
                    if not time_past is None:
                        decay_values, token_counts = self.updateVectors(decay_values=decay_values,
                                                                        last_activation=last_activation,
                                                                        token_counts=token_counts,
                                                                        activated_places=activated_places,
                                                                        current_time=time_recent)

                        if trace[event_id]['concept:name'] == 'discharge':
                            # print('[CAD-HF] discharge event for:', trace.attributes['concept:name'])
                            next_event = self.findNextEventId(event_id, trace)

                            # if next_event_ok:
                            #     # next_event = trace[next_event_id][self.activity_key]
                            #     next_event = label
                            # else:
                            #     print('[CAD-HF] ERROR: NO admission event found after discharge. Not considering this discharge')
                            #     next_event = None

                            if count_resources:
                                timedstatesample = TimedStateSample(time_delta_seconds(init_time, time_recent),
                                                                    copy(decay_values), copy(token_counts), copy(marking),
                                                                    copy(self.place_list),
                                                                    resource_count=copy(resource_count),
                                                                    resource_indices=log_wrapper.getResourceKeys())
                            else:
                                timedstatesample = TimedStateSample(time_delta_seconds(init_time, time_recent),
                                                                    copy(decay_values), copy(token_counts), copy(marking),
                                                                    copy(self.place_list))
                            timedstatesample.setNextEvent(next_event)

                            # try:
                            #     if event["PREDICT"] == "PREDICT" or event["PREDICT"] is True or event["PREDICT"] == "True":
                            #         # TODO: [Maryam]: Is 'label' needs to be added to timedstatesample???
                            #
                            #         print("Encountered a PREDICT=TRUE")
                            #         # if trace.attributes["gender"] == "M":
                            #         #     timedstatesample.setGender(1)
                            #         # else:
                            #         #     timedstatesample.setGender(0)
                            #         # timedstatesample.setAge(float(trace.attributes["age"]))
                            #         # timedstatesample.setEthnicity(trace.attributes["ethnicity"])
                            #         # timedstatesample.setElixhauser(trace.attributes["elixhauser"])
                            #         # timedstatesample.setCharlson(trace.attributes["charlson"])
                            #
                            #         # TODO: [Maryam]: Need to add 'SEVERITY SCORES' here.
                            #
                            #         # tss.append(timedstatesample.export())
                            #         # tss_objs.append(timedstatesample)
                            # except:
                            #     pass

                            if trace.attributes["gender"] == "M":
                                timedstatesample.setGender(int(1))
                            else:
                                timedstatesample.setGender(int(0))
                            timedstatesample.setAge(float(trace.attributes["age"]))
                            timedstatesample.setEthnicity(trace.attributes["ethnicity"])

                            # TODO [Maryam]: Follwing 2 lines need to be changed to accommodate severity score list
                            # timedstatesample.setElixhauser(trace.attributes["elixhauser"])
                            # timedstatesample.setCharlson(trace.attributes["charlson"])
                            timedstatesample.setSeverity(self.get_severity_score(trace.attributes["concept:name"]))

                            tss.append(timedstatesample.export())
                            tss_objs.append(timedstatesample)
        return tss, tss_objs

    def updateVectors(self, decay_values, last_activation, token_counts, activated_places, current_time):
        """ Update Decay Values """
        for place in self.net.places:
            if last_activation[str(place)] == -1:
                decay_values[str(place)] = 0.0
            else:
                delta = time_delta_seconds(last_activation[str(place)], current_time)
                decay_values[str(place)] = self.decay_functions[str(place)].decay(delta)

        """ Update Token Counts """
        for place in activated_places:
            token_counts[str(place)] += 1

        return decay_values, token_counts

    def findNextEventId(self, current_id, trace):
        curr_event = trace[current_id]
        curr_datetime = pd.to_datetime(curr_event[self.timestamp_key])

        next_event_id = current_id + 1

        found = False
        label = "0"
        while next_event_id < len(trace) and not found:
            event = trace[next_event_id]
            # print('[CAD-HF] next event is:', event[self.type_key], 'and', event[self.activity_key])
            if (event[self.activity_key] in self.trans_map.keys()) and (event[self.type_key] == 'admission'):
                # print('[CAD-HF] next event is:',event[self.type_key], 'and', event[self.activity_key])
                # if event[self.type_key] == 'admission':
                found = True
                if event[self.activity_key] == 'unplanned':
                    time_delta = pd.to_datetime(event[self.timestamp_key]) - curr_datetime
                    if time_delta.days < 30:
                        label = "1"
            else:
                next_event_id += 1

        # if next_event_id >= len(trace):
        #     label = False
        # if found == False:
        #     # [Maryam]: If this discharge is the last event for the patient then label should be "False"
        #     #           Means the next admission wad not "within 30 days"
        #     return None, label
        # else:
        return label

    # def findNextEventId(self, current_id, trace):
    # This is the old implementation of findNextEventId
    #     next_event_id = current_id + 1
    #
    #     found = False
    #     while (next_event_id < len(trace) and not found):
    #         event = trace[next_event_id]
    #         if event[self.activity_key] in self.trans_map.keys():
    #             found = True
    #         else:
    #             next_event_id += 1
    #
    #     if found == False:
    #         return None
    #     else:
    #         return next_event_id

    def saveToFile(self, file):
        """
        Save the decay functions of the EnhancedPN to file.
        :param file: Output file
        :return:
        """

        output = dict()
        dumping = copy(self.decay_functions)
        for key in dumping.keys():
            dumping[key] = dumping[key].toJSON()
        output["decayfunctions"] = dumping
        output["resource_keys"] = self.resource_keys
        with open(file, 'w') as fp:
            json.dump(output, fp)

    def loadFromFile(self, file):
        """
        Load decay functions for a given petri net from file.
        :param file: Decay function file
        :return:
        """
        with open(file) as json_file:
            decay_data = json.load(json_file)

        for place in self.net.places:
            self.decay_functions[str(place)] = None

        if not set(decay_data["decayfunctions"].keys()) == set(self.decay_functions.keys()):
            self.decay_functions = {}
            raise ValueError(
                "Set of decay functions is not equal to set of places of the petri net. Was the decay function file build on the same petri net? Loading from file cancelled.")

        for place in decay_data["decayfunctions"].keys():
            DecayFunctionClass = REGISTER[decay_data["decayfunctions"][place]['DecayFunction']]
            df = DecayFunctionClass()
            df.loadFromDict(decay_data["decayfunctions"][place])
            self.decay_functions[str(place)] = df

        self.resource_keys = decay_data["resource_keys"]