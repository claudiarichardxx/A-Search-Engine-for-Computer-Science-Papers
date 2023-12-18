import { Link} from "react-router-dom";
export function ClusterCard({clusterId,clusterName,clusterLength}){

    return(
        <>
        
        <li className="bg-gray-800 px-4 py-4 mt-4 mb-4 rounded text-justify text-white ">
            {clusterName}<br/>
            {clusterLength}
        </li>
        
        </>
    )
}