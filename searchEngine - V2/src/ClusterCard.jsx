import { Link} from "react-router-dom";
export function ClusterCard({clusterId,clusterName,clusterLength}){

    return(
        <>
        <li className=" bg-gray-800 px-4 py-4 rounded text-center text-white ">
            {clusterName}
        </li>
         <h1>{`${clusterLength}`}</h1>
        </>
    )
}